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
        SelfAttentionBlock,
        GatedCrossAttentionBlock,
        RotaryPositionalEncoding,
        AttentionPool,
        CrossAttentionBlock,
    )
except ImportError:
    from .model_utils import (
        MultiScaleSTFTLoss,
        LearnedPositionalEncoding1D,
        SelfAttentionBlock,
        GatedCrossAttentionBlock,
        RotaryPositionalEncoding,
        AttentionPool,
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
    Two-level hierarchical transformer for Encodec latents.

    Level 1 (intra-slot):  Self-attention *within* each segment.
                           Complexity: O(S · T²)   — linear in # slots.
    Level 2 (inter-slot):  Self-attention *across* slot summaries.
                           Complexity: O(S²)        — independent of T.
    Final:                 Target-slot tokens cross-attend to the
                           inter-slot representations.

    Total complexity:  O(S·T² + S²).
    """

    def __init__(
        self,
        latent_dim: int = 128,
        context_length: int = 4,
        forward_context_length: int = 0,
        d_model: int = 256,
        num_heads: int = 8,
        intra_layers: int = 2,
        inter_layers: int = 4,
        final_cross_layers: int = 2,
        ff_mult: int = 4,
        dropout: float = 0.1,
        drop_path: float = 0.1,
        num_layers: int | None = None,
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

        # projection
        self.latent_proj = nn.Sequential(
            nn.Linear(latent_dim, d_model),
            nn.Dropout(dropout),
        )

        # slot-type embedding: 0=past_live, 1=forward_studio, 2=target_studio, 3=unused
        self.slot_type_embed = nn.Embedding(4, d_model)

        # RoPE pos encoding for both levels (separate, since they have different sequence lengths)
        self.intra_pos = RotaryPositionalEncoding(d_model, max_len=1024)
        self.inter_pos = RotaryPositionalEncoding(d_model, max_len=4096)

        # level 1: intra-slot self-attention
        intra_dp = torch.linspace(0, drop_path * 0.5, intra_layers).tolist()
        self.intra_layers = nn.ModuleList(
            [
                SelfAttentionBlock(d_model, num_heads, ff_mult, dropout, intra_dp[i])
                for i in range(intra_layers)
            ]
        )

        self.slot_pool = AttentionPool(d_model)

        # level 2: inter-slot self-attention
        inter_dp = torch.linspace(0, drop_path, inter_layers).tolist()
        self.inter_layers = nn.ModuleList(
            [
                SelfAttentionBlock(d_model, num_heads, ff_mult, dropout, inter_dp[i])
                for i in range(inter_layers)
            ]
        )

        final_dp = torch.linspace(0, drop_path * 0.5, final_cross_layers).tolist()
        self.final_cross = nn.ModuleList(
            [
                CrossAttentionBlock(d_model, num_heads, ff_mult, dropout, final_dp[i])
                for i in range(final_cross_layers)
            ]
        )

        self.output_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, latent_dim),
        )
        nn.init.zeros_(self.output_proj[-1].weight)
        nn.init.zeros_(self.output_proj[-1].bias)

    def _build_slot_types(
        self, batch_size: int, S: int, device: torch.device
    ) -> torch.Tensor:
        """Return (B, S) int tensor: 0=past_live  1=forward_studio  2=target_studio.

        Dynamically assigns types based on actual input size S,
        not just the configured values — so it works even if S
        doesn't exactly match context_length + forward_context_length + 1.
        """
        types = torch.ones(batch_size, S, dtype=torch.long, device=device)
        types[:, -1] = 2  # last slot is always target (studio)

        # everything before context_length is past (live, type 0)
        n_past = min(self.context_length, S - 1)
        types[:, :n_past] = 0  # past slots = live
        # types[:, n_past:-1] = 1 (forward slots = studio, already set above)

        return types

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, S, C, T) — S slots, C=latent_dim, T=temporal frames
        Returns:
            (B, C, T) — predicted latent for the target slot
        """
        B, S, C, T = x.shape

        # raw target for residual connection
        tgt_raw = x[:, -1].transpose(1, 2)  # (B, T, C)

        # project all tokens
        tokens = rearrange(x, "b s c t -> (b s) t c")  # (B*S, T, C)
        tokens = self.latent_proj(tokens)  # (B*S, T, D)

        # level 1: intra-slot self-attention  O(S · T²)
        tokens = self.intra_pos(tokens)
        for layer in self.intra_layers:
            tokens = layer(tokens)  # (B*S, T, D)

        all_tokens = rearrange(tokens, "(b s) t d -> b s t d", b=B)
        tgt_tokens = all_tokens[:, -1]  # (B, T, D)

        # slot pooling: (B*S, T, D) → (B, S, D)
        slot_summaries = self.slot_pool(tokens)  # (B*S, D)
        slot_summaries = rearrange(slot_summaries, "(b s) d -> b s d", b=B)

        # slot-type embeddings
        slot_types = self._build_slot_types(B, S, x.device)
        slot_summaries = slot_summaries + self.slot_type_embed(slot_types)

        # level 2: inter-slot self-attention  O(S²)
        slot_summaries = self.inter_pos(slot_summaries)
        for layer in self.inter_layers:
            slot_summaries = layer(slot_summaries)  # (B, S, D)

        # tokens cross-attend to slot context
        for layer in self.final_cross:
            tgt_tokens = layer(tgt_tokens, slot_summaries)  # (B, T, D)

        delta = self.output_proj(tgt_tokens)  # (B, T, C)
        out = tgt_raw + delta
        return out.clamp(-10.0, 10.0).transpose(1, 2)  # (B, C, T)


class EncodecLatentLightningModule(pl.LightningModule):
    def __init__(
        self,
        model: EncodecLatentModel,
        learning_rate: float = 3e-4,
        sample_rate: int = 48000,
        encodec_bandwidth: float = 6.0,
        encodec_sample_rate: int = 24000,
        cache_dir: str = "logs/encodec_latents",
        forward_context_length: int = 0,
        context_mask_prob: float = 0.2,
    ):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.data_sample_rate = sample_rate
        self.encodec_bandwidth = encodec_bandwidth
        self.encodec_sample_rate = encodec_sample_rate
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

        if S > 2 and self.context_mask_prob > 0:
            mask = (
                torch.rand(B, S - 1, 1, 1, device=x_lat.device) > self.context_mask_prob
            ).float()
            x_lat[:, :-1] = x_lat[:, :-1] * mask

        noise_scale = x_lat.std() * 0.02
        x_lat = x_lat + torch.randn_like(x_lat) * noise_scale

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

    def compute_loss(self, pred, target, decode_loss=False):
        pred, target = self._align_time(pred, target)

        l1 = F.l1_loss(pred, target)
        cos = (
            1.0
            - F.cosine_similarity(
                pred.transpose(1, 2), target.transpose(1, 2), dim=-1
            ).mean()
        )
        loss = l1 + 0.1 * cos

        return loss

    def training_step(self, batch, batch_idx):
        studio_audio, live_audio = batch["studio_audio"], batch["live_audio"]
        cache_keys = batch.get("cache_key")

        x_studio = self._encode_audio(
            studio_audio, [f"studio::{k}" for k in cache_keys] if cache_keys else None
        )
        x_live = self._encode_audio(
            live_audio, [f"live::{k}" for k in cache_keys] if cache_keys else None
        )

        # ── Build mixed input: past=LIVE, forward+target=STUDIO ──
        ctx_len = self.model.context_length
        mixed = x_studio.clone()
        mixed[:, :ctx_len] = x_live[:, :ctx_len]  # past context uses live audio

        mixed = self._augment_latents(mixed)

        target = x_live[:, -1]
        y_pred = self(mixed)

        # decode every 2 steps to save VRAM
        decode_loss = self.global_step % 2 == 0
        loss = self.compute_loss(y_pred, target, decode_loss=decode_loss)

        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        studio_audio, live_audio = batch["studio_audio"], batch["live_audio"]
        cache_keys = batch.get("cache_key")

        x_studio = self._encode_audio(
            studio_audio, [f"studio::{k}" for k in cache_keys] if cache_keys else None
        )
        x_live = self._encode_audio(
            live_audio, [f"live::{k}" for k in cache_keys] if cache_keys else None
        )

        # Same mixed input strategy
        ctx_len = self.model.context_length
        mixed = x_studio.clone()
        mixed[:, :ctx_len] = x_live[:, :ctx_len]

        y_pred = self(mixed)
        target = x_live[:, -1]

        # ALWAYS decode on val so metric is meaningful
        loss = self.compute_loss(y_pred, target, decode_loss=True)
        trivial = self.compute_loss(x_studio[:, -1], target, decode_loss=False)

        self.log("val/loss", loss, prog_bar=True)
        self.log("val/trivial_baseline", trivial)

        if not self._val_preview_logged:
            self._val_preview_logged = True
            self._log_latent_preview(y_pred.detach(), target.detach())
            self._log_audio_preview(y_pred.detach(), target.detach())
        return loss

    def _log_audio_preview(self, pred, target):
        """Decode and log actual audio so you can HEAR the quality."""
        if pred.numel() == 0:
            return
        with torch.no_grad():
            pred_audio = self.encodec.decoder(pred[:1].float())
            target_audio = self.encodec.decoder(target[:1].float())
        logger = getattr(self.logger, "experiment", None)
        if logger and hasattr(logger, "log"):
            import wandb as wb

            logger.log(
                {
                    "val/pred_audio": wb.Audio(
                        pred_audio[0, 0].cpu().float().numpy(),
                        sample_rate=self.encodec_sample_rate,
                        caption="predicted",
                    ),
                    "val/target_audio": wb.Audio(
                        target_audio[0, 0].cpu().float().numpy(),
                        sample_rate=self.encodec_sample_rate,
                        caption="target",
                    ),
                }
            )

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

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            [p for p in self.parameters() if p.requires_grad],
            lr=self.learning_rate,
            betas=(0.9, 0.98),
            weight_decay=1e-6,
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=150,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "monitor": "train/loss",
                "frequency": 1,
            },
        }


def main():
    from torchinfo import summary

    model = EncodecLatentModel(
        latent_dim=128,
        context_length=12,
        forward_context_length=12,
        d_model=128,
        num_heads=4,
        num_layers=2,
        ff_mult=2,
        dropout=0.2,
        drop_path=0.1,
    )
    summary(model, input_size=(1, 9, 128, 64))


if __name__ == "__main__":
    main()
