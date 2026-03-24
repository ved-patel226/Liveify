import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchaudio
import hashlib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple
from einops import rearrange
from encodec import EncodecModel

try:
    from model_utils import (
        MultiScaleSTFTLoss,
        LearnedPositionalEncoding1D,
        CrossAttentionBlock,
    )
except ImportError:
    from .model_utils import (
        MultiScaleSTFTLoss,
        LearnedPositionalEncoding1D,
        CrossAttentionBlock,
    )

matplotlib.use("Agg")


class BaseLiveifyModel(torch.nn.Module):
    def __init__(
        self,
        input_fdim=128,
        input_tdim=1024,
        in_channels=1,
        out_channels=1,
        context_length=4,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.input_fdim = input_fdim
        self.input_tdim = input_tdim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.context_length = context_length
        self.num_slots = context_length + 1

    def forward(self, x):
        raise NotImplementedError


class EncodecLatentModel(BaseLiveifyModel):
    """
    v2: ~500K params (down from 6.5M).
    """

    def __init__(
        self,
        latent_dim: int = 128,
        context_length: int = 4,
        forward_context_length: int = 0,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        ff_mult: int = 2,
        dropout: float = 0.3,
        drop_path: float = 0.1,
    ):
        super().__init__(
            input_fdim=latent_dim,
            input_tdim=1,
            in_channels=1,
            out_channels=1,
            context_length=context_length,
        )
        self.latent_dim = latent_dim
        self.context_length = context_length
        self.forward_context_length = forward_context_length
        self.d_model = d_model
        self.num_slots = context_length + 1 + forward_context_length

        # ═══ CHANGE: single shared projection (was two separate ones) ═══
        self.latent_proj = nn.Sequential(
            nn.Linear(latent_dim, d_model),
            nn.Dropout(dropout),
        )

        # ═══ NEW: slot positional embeddings ═══
        # Without this, the model has NO idea which context frame is recent vs old
        self.slot_embed = nn.Embedding(self.num_slots, d_model)

        # ═══ NEW: temporal positional encoding within each segment ═══
        self.temporal_pos = LearnedPositionalEncoding1D(
            d_model, max_len=512, dropout=dropout
        )

        # Linearly increasing stochastic depth rates
        dp_rates = torch.linspace(0, drop_path, num_layers).tolist()

        self.layers = nn.ModuleList(
            [
                CrossAttentionBlock(d_model, num_heads, ff_mult, dropout, dp_rates[i])
                for i in range(num_layers)
            ]
        )

        self.output_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, latent_dim),
        )

        # Zero-init → starts as identity (residual = 0)
        nn.init.zeros_(self.output_proj[-1].weight)
        nn.init.zeros_(self.output_proj[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, C, T = x.shape

        ctx = x[:, :-1]  # (B, S-1, C, T)
        tgt = x[:, -1]  # (B, C, T)
        tgt_raw = rearrange(tgt, "b c t -> b t c")  # (B, T, latent_dim)

        # ═══ Build context: project each slot, add temporal + slot position ═══
        ctx_tokens = []
        for s in range(S - 1):
            slot = rearrange(ctx[:, s], "b c t -> b t c")  # (B, T, C)
            proj = self.latent_proj(slot)  # (B, T, d_model)
            proj = self.temporal_pos(proj)  # temporal position within segment
            proj = proj + self.slot_embed.weight[s]  # which segment in the window
            ctx_tokens.append(proj)
        ctx_proj = torch.cat(ctx_tokens, dim=1)  # (B, (S-1)*T, d_model)

        # ═══ Target: same projection + its own slot identity ═══
        tgt_proj = self.latent_proj(tgt_raw)
        tgt_proj = self.temporal_pos(tgt_proj)
        tgt_proj = tgt_proj + self.slot_embed.weight[S - 1]

        for layer in self.layers:
            tgt_proj = layer(tgt_proj, ctx_proj)

        delta = self.output_proj(tgt_proj)  # (B, T, latent_dim)
        out = tgt_raw + delta
        return out.clamp(-10.0, 10.0).transpose(1, 2)  # (B, latent_dim, T)


class EncodecLatentLightningModule(pl.LightningModule):
    def __init__(
        self,
        model: EncodecLatentModel,
        learning_rate: float = 3e-4,
        sample_rate: int = 48000,
        encodec_bandwidth: float = 6.0,
        encodec_sample_rate: int = 24000,
        cache_dir: str = "logs/encodec_latents",
        latent_noise_std: float = 0.05,  # ← increased from 0.02
        forward_context_length: int = 0,
        context_mask_prob: float = 0.2,
    ):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.data_sample_rate = sample_rate
        self.encodec_bandwidth = encodec_bandwidth
        self.encodec_sample_rate = encodec_sample_rate
        self.latent_noise_std = latent_noise_std
        self.forward_context_length = forward_context_length
        self.context_mask_prob = context_mask_prob

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._warned_sr_mismatch = False
        self._val_preview_logged = False

        self.encodec = EncodecModel.encodec_model_24khz()
        self.encodec.set_target_bandwidth(self.encodec_bandwidth)
        self.encodec.eval()
        for p in self.encodec.parameters():
            p.requires_grad = False

        self.loss_l1 = nn.L1Loss()
        self.spectral_loss = MultiScaleSTFTLoss()
        self.save_hyperparameters(ignore=["model", "encodec"])

    def _augment_latents(self, x_lat: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return x_lat
        B, S, C, T = x_lat.shape
        x_lat = x_lat.clone()

        # Context masking (increased)
        if S > 2 and self.context_mask_prob > 0:
            mask = (
                torch.rand(B, S - 1, 1, 1, device=x_lat.device) > 0.4  # ← was 0.2
            ).float()
            x_lat[:, :-1] = x_lat[:, :-1] * mask

        noise_scale = x_lat.std() * 0.15  # ← was 0.05
        x_lat = x_lat + torch.randn_like(x_lat) * noise_scale

        gain = 1.0 + 0.2 * torch.randn(B, S, 1, 1, device=x_lat.device)  # ← was 0.1
        x_lat = x_lat * gain

        time_mask = (torch.rand(B, 1, 1, T, device=x_lat.device) > 0.1).float()
        x_lat = x_lat * time_mask

        chan_mask = (torch.rand(B, 1, C, 1, device=x_lat.device) > 0.1).float()
        x_lat = x_lat * chan_mask

        return x_lat

    def _mixup(self, x_lat, y_lat, alpha=0.3):
        """Interpolate between random pairs — effectively infinite training examples."""
        if not self.training or x_lat.size(0) < 2:
            return x_lat, y_lat
        lam = torch.distributions.Beta(alpha, alpha).sample().to(x_lat.device)
        idx = torch.randperm(x_lat.size(0), device=x_lat.device)
        x_lat = lam * x_lat + (1 - lam) * x_lat[idx]
        y_lat = lam * y_lat + (1 - lam) * y_lat[idx]
        return x_lat, y_lat

    def _encode_audio(self, audio, cache_keys=None):
        B, S, L = audio.shape
        cache_paths = [None] * B
        latents = [None] * B
        missing_indices = []

        if cache_keys is not None:
            for i, key in enumerate(cache_keys):
                # ═══ FIX: include shape in hash so config changes invalidate cache ═══
                full_key = f"{key}::S{S}::L{L}"
                path = (
                    self.cache_dir / f"{hashlib.sha1(full_key.encode()).hexdigest()}.pt"
                )
                cache_paths[i] = path
                if path.exists():
                    cached = torch.load(path, map_location=self.device)
                    # Validate shape matches current config
                    if cached.shape[0] == S:
                        latents[i] = cached
                    else:
                        missing_indices.append(i)  # shape mismatch → recompute
                else:
                    missing_indices.append(i)
        else:
            missing_indices = list(range(B))

        if missing_indices:
            wav = audio[missing_indices].view(len(missing_indices) * S, 1, L)
            if self.data_sample_rate != self.encodec_sample_rate:
                if not self._warned_sr_mismatch:
                    self._warned_sr_mismatch = True
                    print(
                        f"[warning] Resampling {self.data_sample_rate}→{self.encodec_sample_rate}"
                    )
                wav = torchaudio.functional.resample(
                    wav, self.data_sample_rate, self.encodec_sample_rate
                )

            # Ensure uniform audio length after resampling (resampling can cause length variations)
            max_audio_len = wav.shape[-1]
            if wav.shape[-1] < max_audio_len:
                pad_amount = max_audio_len - wav.shape[-1]
                wav = F.pad(wav, (0, pad_amount))

            wav = wav.to(self.device)
            wav = wav / wav.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)

            with torch.no_grad():
                lat_batch = self.encodec.encoder(wav)

            lat_batch = lat_batch.view(len(missing_indices), S, *lat_batch.shape[1:])

            for idx, si in enumerate(missing_indices):
                latents[si] = lat_batch[idx]
                if cache_paths[si] is not None:
                    torch.save(lat_batch[idx].cpu(), cache_paths[si])

        # Ensure all latents have the same temporal dimension before stacking
        if latents and any(lat is not None for lat in latents):
            max_temporal_dim = 0
            for lat in latents:
                if lat is not None:
                    max_temporal_dim = max(max_temporal_dim, lat.shape[-1])

            if max_temporal_dim > 0:
                padded_latents = []
                for lat in latents:
                    if lat is not None:
                        # Pad the temporal dimension (last dimension)
                        pad_amount = max_temporal_dim - lat.shape[-1]
                        if pad_amount > 0:
                            lat = F.pad(lat, (0, pad_amount))
                        padded_latents.append(lat)
                    else:
                        padded_latents.append(lat)
                latents = padded_latents

        return torch.stack(latents).to(self.device)

    def forward(self, x):
        return self.model(x)

    def _align_time(self, pred, target):
        pt, tt = pred.shape[-1], target.shape[-1]
        if pt > tt:
            pred = pred[..., :tt]
        elif tt > pt:
            target = target[..., :pt]
        return pred, target

    def compute_loss(self, pred, target, **kwargs):
        pred, target = self._align_time(pred, target)
        l1 = F.l1_loss(pred, target)
        cos = (
            1.0
            - F.cosine_similarity(
                pred.transpose(1, 2), target.transpose(1, 2), dim=-1
            ).mean()
        )
        return l1 + 0.1 * cos

    def training_step(self, batch, batch_idx):
        studio_audio, live_audio = batch["studio_audio"], batch["live_audio"]
        cache_keys = batch.get("cache_key")

        x_lat = self._encode_audio(
            studio_audio, [f"studio::{k}" for k in cache_keys] if cache_keys else None
        )
        y_lat = self._encode_audio(
            live_audio, [f"live::{k}" for k in cache_keys] if cache_keys else None
        )

        x_lat = self._augment_latents(x_lat)  # keep augmentation
        # NO mixup, NO R-drop

        target = y_lat[:, -1]
        y_pred = self(x_lat)
        loss = self.compute_loss(y_pred, target)

        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        studio_audio, live_audio = batch["studio_audio"], batch["live_audio"]
        cache_keys = batch.get("cache_key")

        x_lat = self._encode_audio(
            studio_audio, [f"studio::{k}" for k in cache_keys] if cache_keys else None
        )
        y_lat = self._encode_audio(
            live_audio, [f"live::{k}" for k in cache_keys] if cache_keys else None
        )

        y_pred = self(x_lat)
        target = y_lat[:, -1]
        loss = self.compute_loss(y_pred, target)
        trivial = self.compute_loss(x_lat[:, -1], target)

        self.log("val/loss", loss, prog_bar=True)
        self.log("val/trivial_baseline", trivial)

        if not self._val_preview_logged:
            self._val_preview_logged = True
            self._log_latent_preview(y_pred.detach(), target.detach())
        return loss

    def on_validation_epoch_start(self):
        self._val_preview_logged = False

    def _log_latent_preview(self, pred, target):
        if pred.numel() == 0:
            return
        p, t = pred[0].cpu().float().mean(0), target[0].cpu().float().mean(0)
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(p.numpy(), label="pred", alpha=0.8)
        ax.plot(t.numpy(), label="target", alpha=0.8)
        ax.set_title("Latent mean trace")
        ax.legend()
        fig.tight_layout()
        logger = getattr(self.logger, "experiment", None)
        if logger:
            try:
                if hasattr(logger, "add_figure"):
                    logger.add_figure("val/latent_trace", fig, self.current_epoch)
                elif hasattr(logger, "log"):
                    logger.log({"val/latent_trace": fig})
            finally:
                pass
        plt.close(fig)

    def on_before_optimizer_step(self, optimizer):
        if self.global_step % 10 == 0:
            norm = torch.nn.utils.clip_grad_norm_(
                [p for p in self.parameters() if p.requires_grad], float("inf"), 2
            )
            self.log("train/grad_norm", norm, on_step=True, on_epoch=False)

    # ═══ CHANGE: cosine LR with warmup (was ReduceLROnPlateau with patience=1000) ═══
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            [p for p in self.parameters() if p.requires_grad],
            lr=self.learning_rate,
            betas=(0.9, 0.98),
            weight_decay=0.1,  # ← was 0.05
        )

        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = min(500, total_steps // 10)

        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }


def main():
    from torchinfo import summary

    model = EncodecLatentModel(
        latent_dim=128,
        context_length=4,
        forward_context_length=4,
        d_model=128,  # ← was 256
        num_heads=4,  # ← was 8
        num_layers=2,  # default now 2
        ff_mult=2,  # ← was 4
        dropout=0.2,  # default now 0.3
        drop_path=0.1,
    )
    summary(model, input_size=(1, 5, 128, 64))


if __name__ == "__main__":
    main()
