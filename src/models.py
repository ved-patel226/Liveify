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
    def __init__(
        self,
        latent_dim: int = 128,
        context_length: int = 4,
        forward_context_length: int = 4,
        d_model: int = 384,
        num_heads: int = 6,
        num_layers: int = 8,
        ff_mult: int = 4,
        dropout: float = 0.2,
        drop_path: float = 0.05,
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

        self.latent_proj = nn.Sequential(
            nn.Linear(latent_dim, d_model),
            nn.Dropout(dropout),
        )

        # past=0, current=1, future=2
        self.slot_type_embed = nn.Embedding(3, d_model)

        self.pos_enc = RotaryPositionalEncoding(d_model, max_len=4096)

        dp_rates = torch.linspace(0, drop_path, num_layers).tolist()
        self.blocks = nn.ModuleList(
            [
                SelfAttentionBlock(d_model, num_heads, ff_mult, dropout, dp_rates[i])
                for i in range(num_layers)
            ]
        )

        self.output_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, latent_dim),
        )
        nn.init.zeros_(self.output_proj[-1].weight)
        nn.init.zeros_(self.output_proj[-1].bias)

    def _build_slot_types(self, batch_size, S, device):
        types = torch.zeros(batch_size, S, dtype=torch.long, device=device)  # 0 = past
        types[:, -1] = 1  # last slot = target
        n_past = min(self.context_length, S - 1)
        if S > n_past + 1:
            types[:, n_past:-1] = 2  # future context
        return types

    def forward(self, x):
        B, S, C, T = x.shape
        target_idx = self.context_length
        tgt_raw = x[:, -1].transpose(1, 2)  # (B, T, C) last slot is target

        tokens = rearrange(x, "b s c t -> b (s t) c")
        tokens = self.latent_proj(tokens)  # (B, S*T, d_model)

        slot_types = self._build_slot_types(B, S, x.device)  # (B, S)
        slot_types = slot_types.unsqueeze(-1).expand(-1, -1, T)  # (B, S, T)
        slot_types = rearrange(slot_types, "b s t -> b (s t)")  # (B, S*T)
        tokens = tokens + self.slot_type_embed(slot_types)

        tokens = self.pos_enc(tokens)

        for block in self.blocks:
            tokens = block(tokens)

        tgt_tokens = tokens[:, -T:]  # (B, T, d_model)
        delta = self.output_proj(tgt_tokens)  # (B, T, C)
        out = tgt_raw + delta
        return out.clamp(-10.0, 10.0).transpose(1, 2)  # (B, C, T)

    def generate_autoregressive(
        self,
        studio_context: torch.Tensor,  # (B, ctx_len, C, T)
        n_steps: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        decode_strategy: str = "sample",  # "sample", "argmax", "deterministic"
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Autoregressive generation using model's own outputs as future context.

        Args:
            studio_context: Initial studio latents (B, context_length, C, T)
            n_steps: Number of future steps to generate
            temperature: Sampling temperature (higher = more diverse)
            top_k: Top-k filtering
            top_p: Nucleus sampling threshold
            decode_strategy: "sample", "argmax", or "deterministic"

        Returns:
            all_outputs: All generated tokens (B, n_steps, C, T)
            all_hidden: Hidden states at each step for potential inspection
        """
        B, ctx_len, C, T = studio_context.shape
        device = studio_context.device

        total_slots = ctx_len + n_steps

        context = torch.zeros(B, total_slots, C, T, device=device)
        context[:, :ctx_len] = studio_context

        all_outputs = []

        with torch.no_grad():  # dont use gradients for gen
            # TODO: mabye benefical to allow gradients? e.g. for scheduled sampling or other gen-time strategies
            for step in range(n_steps):
                current_slot = ctx_len + step

                slot_types = self._build_slot_types(
                    B, total_slots, device, generation_mode=True
                )

                output = self._forward_partial(context, slot_types, current_slot)

                decoded = self._decode_step(
                    output, temperature, top_k, top_p, decode_strategy
                )

                all_outputs.append(decoded)

                context[:, current_slot] = decoded

        all_outputs = torch.stack(all_outputs, dim=1)

        return all_outputs

    def _forward_partial(
        self,
        context: torch.Tensor,
        slot_types: torch.Tensor,
        target_slot: int,
    ) -> torch.Tensor:
        """Forward pass with partial context, predicting only target_slot."""
        B, S, C, T = context.shape

        tokens = rearrange(context, "b s c t -> (b s) t c")
        tokens = self.latent_proj(tokens)

        # intra-slot attention with positional encoding
        tokens = self.intra_pos(tokens)
        for layer in self.intra_layers:
            tokens = layer(tokens)

        all_tokens = rearrange(tokens, "(b s) t d -> b s t d", b=B)

        slot_summaries = self.slot_pool(tokens)
        slot_summaries = rearrange(slot_summaries, "(b s) d -> b s d", b=B)
        slot_summaries = slot_summaries + self.slot_type_embed(slot_types)

        slot_summaries = self.inter_pos(slot_summaries)
        for layer in self.inter_layers:
            slot_summaries = layer(slot_summaries)

        tgt_tokens = all_tokens[:, target_slot, 0]

        for layer in self.final_cross:
            tgt_tokens = layer(tgt_tokens, slot_summaries)

        delta = self.output_proj(tgt_tokens)

        return delta

    def _decode_step(
        self,
        delta: torch.Tensor,
        temperature: float,
        top_k: Optional[int],
        top_p: Optional[float],
        strategy: str,
    ) -> torch.Tensor:
        """Apply decoding strategy to get final output."""
        if strategy == "argmax":
            return delta.clamp(-10.0, 10.0)

        elif strategy == "sample":
            if temperature != 1.0:
                delta = delta / temperature

            if top_k is not None and top_k > 0:
                top_k_vals, _ = torch.topk(delta, min(top_k, delta.shape[-1]), dim=-1)
                threshold = top_k_vals[..., -1:]
                delta = torch.where(
                    delta < threshold,
                    torch.full_like(delta, float("-inf")),
                    delta,
                )

            if top_p is not None and top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(delta, descending=True)
                cum_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

                sorted_mask = cum_probs > top_p
                sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
                sorted_mask[..., 0] = False

                indices_to_remove = sorted_mask.scatter(1, sorted_indices, sorted_mask)
                delta = torch.where(
                    indices_to_remove,
                    torch.full_like(delta, float("-inf")),
                    delta,
                )

            noise = torch.randn_like(delta) * (temperature * 0.1)
            return (delta + noise).clamp(-10.0, 10.0)

        else:  # deterministic
            return delta.clamp(-10.0, 10.0)


class EncodecLatentLightningModule(pl.LightningModule):
    def __init__(
        self,
        model: EncodecLatentModel,
        learning_rate: float = None,
        sample_rate: int = 48000,
        encodec_bandwidth: float = 6.0,
        encodec_sample_rate: int = 24000,
        cache_dir: str = "logs/encodec_latents",
        forward_context_length: int = 0,
        context_mask_prob: float = 0.2,
        batch_size: int = None,
        accumulate_grad_batches: int = None,
    ):
        super().__init__()
        self.model = model

        if learning_rate is None:
            learning_rate = 3e-4 * (384 / model.d_model) ** 0.5
        self.learning_rate = learning_rate

        if batch_size is None:
            batch_size = 256 * (model.d_model // 384)
        if accumulate_grad_batches is None:
            accumulate_grad_batches = max(1, model.d_model // 384)

        self.batch_size = batch_size
        self.accumulate_grad_batches = accumulate_grad_batches
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

    # ─── latent-level augmentation (works on both paths) ───

    def _augment_latents(self, x_lat):
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

    def _encode_audio(self, audio, cache_keys=None):
        B, S, L = audio.shape
        cache_paths = [None] * B
        latents = [None] * B
        missing_indices = []

        if cache_keys is not None:
            for i, key in enumerate(cache_keys):
                full_key = f"{key}::S{S}::L{L}"
                path = (
                    self.cache_dir / f"{hashlib.sha1(full_key.encode()).hexdigest()}.pt"
                )
                cache_paths[i] = path
                if path.exists():
                    cached = torch.load(path, map_location=self.device)
                    if cached.shape[0] == S:
                        latents[i] = cached
                    else:
                        missing_indices.append(i)
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

            max_audio_len = wav.shape[-1]
            if wav.shape[-1] < max_audio_len:
                wav = F.pad(wav, (0, max_audio_len - wav.shape[-1]))

            wav = wav.to(self.device)
            wav = wav / wav.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)

            with torch.no_grad():
                lat_batch = self.encodec.encoder(wav)

            lat_batch = lat_batch.view(len(missing_indices), S, *lat_batch.shape[1:])

            for idx, si in enumerate(missing_indices):
                latents[si] = lat_batch[idx]
                if cache_paths[si] is not None:
                    torch.save(lat_batch[idx].cpu(), cache_paths[si])

        if latents and any(lat is not None for lat in latents):
            max_temporal_dim = max(lat.shape[-1] for lat in latents if lat is not None)
            padded = []
            for lat in latents:
                if lat is not None:
                    pad_amount = max_temporal_dim - lat.shape[-1]
                    if pad_amount > 0:
                        lat = F.pad(lat, (0, pad_amount))
                padded.append(lat)
            latents = padded

        return torch.stack(latents).to(self.device)

    def _get_latents(self, batch):
        """Return (x_studio, x_live) regardless of whether the batch
        contains precomputed latents or raw audio."""
        if "studio_latents" in batch:
            return (
                batch["studio_latents"].to(self.device),
                batch["live_latents"].to(self.device),
            )
        studio_audio = batch["studio_audio"]
        live_audio = batch["live_audio"]
        cache_keys = batch.get("cache_key")
        x_studio = self._encode_audio(
            studio_audio,
            [f"studio::{k}" for k in cache_keys] if cache_keys else None,
        )
        x_live = self._encode_audio(
            live_audio,
            [f"live::{k}" for k in cache_keys] if cache_keys else None,
        )
        return x_studio, x_live

    # ─── training / validation ───

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

        if decode_loss:
            with torch.no_grad():
                pred_audio = self.encodec.decoder(pred)
                target_audio = self.encodec.decoder(target)
            stft = self.spectral_loss(pred_audio, target_audio)
            loss = loss + 0.1 * stft

        return loss

    def training_step(self, batch, batch_idx):
        x_studio, x_live = self._get_latents(batch)
        batch_size = x_studio.shape[0]

        ctx_len = self.model.context_length
        mixed = x_studio.clone()
        mixed[:, :ctx_len] = x_live[:, :ctx_len]

        mixed = self._augment_latents(mixed)

        target = x_live[:, -1]
        y_pred = self(mixed)

        loss = self.compute_loss(y_pred, target, decode_loss=True)

        scaled_loss = loss / self.accumulate_grad_batches

        self.log("train/loss", loss, prog_bar=True, batch_size=batch_size)
        return scaled_loss

    def validation_step(self, batch, batch_idx):
        x_studio, x_live = self._get_latents(batch)
        batch_size = x_studio.shape[0]

        ctx_len = self.model.context_length
        mixed = x_studio.clone()
        mixed[:, :ctx_len] = x_live[:, :ctx_len]

        y_pred = self(mixed)
        target = x_live[:, -1]

        loss = self.compute_loss(y_pred, target, decode_loss=True)
        trivial = self.compute_loss(x_studio[:, -1], target, decode_loss=False)

        self.log("val/loss", loss, prog_bar=True, batch_size=batch_size)
        self.log("val/trivial_baseline", trivial, batch_size=batch_size)

        if not self._val_preview_logged:
            self._val_preview_logged = True
            self._log_latent_preview(y_pred.detach(), target.detach())
            self._log_audio_preview(y_pred.detach(), target.detach())
        return loss

    def _log_audio_preview(self, pred, target):
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
        p = pred[0].cpu().float().mean(0)
        t = target[0].cpu().float().mean(0)
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
            factor=0.7,  # 0.5 → 0.7 (gentler reduction)
            patience=150,  # 150 → 50 (faster response)
            min_lr=1e-7,  # don't go below this
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

    @torch.no_grad()
    def generate(
        self,
        studio_audio: Optional[torch.Tensor] = None,
        studio_latents: Optional[torch.Tensor] = None,
        n_steps: int = 24,
        temperature: float = 1.0,
        decode_strategy: str = "sample",
        initial_live_context: Optional[torch.Tensor] = None,
        return_audio: bool = True,
    ) -> dict:
        """Generate live audio from studio audio autoregressively.

        Args:
            studio_audio: Raw studio audio (B, L) - will be encoded if provided
            studio_latents: Pre-encoded studio latents (B, S, C, T)
            n_steps: Number of steps to generate
            temperature: Sampling temperature
            decode_strategy: "sample", "argmax", or "deterministic"
            initial_live_context: Optional initial live context (for continuation)
            return_audio: Whether to decode latents to audio

        Returns:
            Dictionary with generated latents and optionally audio
        """
        self.model.eval()

        if studio_latents is not None:
            x_studio = studio_latents.to(self.device)
        elif studio_audio is not None:
            x_studio = self._encode_audio(studio_audio.to(self.device))[0]
        else:
            raise ValueError("Must provide either studio_audio or studio_latents")

        B, S, C, T = x_studio.shape

        if initial_live_context is not None:
            init_context = initial_live_context.to(self.device)
            ctx_len = init_context.shape[1]
            context = torch.cat([init_context, x_studio[:, :n_steps]], dim=1)
        else:
            ctx_len = self.model.context_length
            context = x_studio[:, :ctx_len]

        generated_latents = self.model.generate_autoregressive(
            studio_context=context,
            n_steps=n_steps,
            temperature=temperature,
            decode_strategy=decode_strategy,
        )

        result = {
            "generated_latents": generated_latents,
            "studio_context": context,
        }

        if return_audio:
            all_latents = torch.cat([context, generated_latents], dim=1)
            flat_latents = rearrange(all_latents, "b s c t -> (b s) c t")

            with torch.no_grad():
                audio = self.encodec.decoder(flat_latents.float())

            audio = rearrange(audio, "(b s) c t -> b s c t", b=B)
            result["generated_audio"] = audio[:, ctx_len:]
            result["full_audio"] = audio

        return result

    @torch.no_grad()
    def generate_streaming(
        self,
        studio_chunk: torch.Tensor,
        generated_cache: List[torch.Tensor],
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Streaming generation - add one step at a time.
        Useful for real-time applications.

        Args:
            studio_chunk: New studio chunk (B, C, T)
            generated_cache: List of previously generated latents
            temperature: Sampling temperature

        Returns:
            Next generated latent (B, C, T)
        """
        self.model.eval()

        ctx_len = self.model.context_length
        cache_len = len(generated_cache)

        if cache_len < ctx_len:
            return torch.zeros_like(studio_chunk)

        context = torch.stack(generated_cache[-ctx_len:] + [studio_chunk], dim=1)

        generated = self.model.generate_autoregressive(
            studio_context=context,
            n_steps=1,
            temperature=temperature,
        )

        return generated[:, 0]


def main():
    from torchinfo import summary

    model = EncodecLatentModel(
        latent_dim=128,
        d_model=512,
        num_heads=8,
        num_layers=10,
        ff_mult=4,
        dropout=0.2,
        drop_path=0.05,
    )

    summary(model, input_size=(1, 9, 128, 16))


if __name__ == "__main__":
    main()
