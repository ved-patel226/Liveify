"""Liveify model architectures and Lightning modules."""

import io
import hashlib
from pathlib import Path
from typing import List, Optional, Tuple

import librosa
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import warnings
from einops import rearrange
from encodec import EncodecModel
from model_utils import (
    PatchEmbedding,
    PositionalEncoding,
    PatchReconstruction,
    TransformerEncoderLayer,
    to_2tuple,
)

matplotlib.use("Agg")


class SpectrogramLoss(nn.Module):
    """
    SIMPLIFIED Loss for mel spectrograms.
    Uses L1 (sparse differences) + L2 (overall structure).
    No PSA or SI-SNR since we're working with mel spectrograms, not complex spectrograms.
    """

    def __init__(self, l1_weight: float = 1.0, l2_weight: float = 1.0):
        super().__init__()
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mixture=None):
        loss = 0.0
        if self.l1_weight > 0:
            loss += self.l1_weight * F.l1_loss(pred, target)
        if self.l2_weight > 0:
            loss += self.l2_weight * F.mse_loss(pred, target)
        return loss


def audio_to_mel_tensor(
    audio: torch.Tensor,
    sr: int,
    n_mels: int = 256,
    n_fft: int = 1024,
    hop_length: int = 512,
) -> torch.Tensor:
    """(B, slots, samples) -> (B, slots, 1, n_mels, T)"""
    B, slots, samples = audio.shape
    device = audio.device

    mel_fb = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels, fmax=sr // 2)
    mel_fb = torch.tensor(mel_fb, dtype=torch.float32, device=device)
    window = torch.hann_window(n_fft, device=device)

    specs = []
    for slot_idx in range(slots):
        slot_audio = audio[:, slot_idx, :]
        stft = torch.stft(
            slot_audio,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=n_fft,
            window=window,
            return_complex=True,
        )
        power = stft.abs() ** 2
        mel = torch.matmul(mel_fb, power)
        mel_db = 10.0 * torch.log10(mel.clamp(min=1e-9))
        mel_min = mel_db.amin(dim=(-2, -1), keepdim=True)
        mel_max = mel_db.amax(dim=(-2, -1), keepdim=True)
        mel_db = 2.0 * (mel_db - mel_min) / (mel_max - mel_min + 1e-8) - 1.0
        specs.append(mel_db.unsqueeze(1))

    return torch.stack(specs, dim=1)


def mel_to_audio_griffin_lim(
    mel_spec: np.ndarray,
    sr: int,
    n_fft: int = 1024,
    hop_length: int = 512,
    n_iter: int = 64,
) -> np.ndarray:
    """Approximate inversion of a normalised mel spectrogram for quick audio previews."""
    mel_db = (mel_spec + 1.0) * 40.0 - 80.0
    mel_power = np.power(10.0, mel_db / 10.0)

    n_mels = mel_spec.shape[0]
    mel_fb = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)
    mel_fb_inv = np.linalg.pinv(mel_fb)
    linear_power = np.maximum(mel_fb_inv @ mel_power, 0.0)
    linear_mag = np.sqrt(linear_power)

    audio = librosa.griffinlim(
        linear_mag, n_iter=n_iter, hop_length=hop_length, win_length=n_fft
    )
    peak = np.abs(audio).max()
    if peak > 0:
        audio = audio / peak
    return audio.astype(np.float32)


def _fig_to_numpy(fig) -> np.ndarray:
    """Render a matplotlib figure to an (H, W, 3) uint8 array."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    buf.seek(0)
    from PIL import Image

    img = np.array(Image.open(buf).convert("RGB"))
    buf.close()
    return img


class LearnedPositionalEncoding1D(nn.Module):
    """Learned 1D positional encoding with dropout for sequence length L."""

    def __init__(self, embed_dim: int, max_len: int = 4096, dropout: float = 0.1):
        super().__init__()
        self.pos = nn.Parameter(torch.zeros(1, max_len, embed_dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, embed_dim)
        Returns:
            (batch, seq_len, embed_dim) with positional encodings added
        """
        L = x.shape[1]
        x = x + self.pos[:, :L, :]
        return self.dropout(x)


class BaseLiveifyModel(torch.nn.Module):
    """
    Base class for Liveify models.
    Provides common initialization, forward logic, and optimizer setup.
    Subclasses should implement specific architectures.
    """

    def __init__(
        self,
        input_fdim: int = 128,
        input_tdim: int = 1024,
        in_channels: int = 1,
        out_channels: int = 1,
        context_length: int = 4,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.input_fdim = input_fdim
        self.input_tdim = input_tdim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.context_length = context_length
        self.num_slots = context_length + 1  # context + current

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass; override in subclass."""
        raise NotImplementedError


class CrossAttentionBlock(nn.Module):
    """Cross-attention block with self-attention, cross-attention, and feedforward."""

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.cross_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, target: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            target: (batch, target_time, d_model)
            context: (batch, context_length, d_model)
        Returns:
            (batch, target_time, d_model)
        """
        # self-attention on target
        h = self.norm1(target)
        target = target + self.self_attn(h, h, h)[0]

        h = self.norm2(target)
        target = target + self.cross_attn(h, context, context)[0]

        target = target + self.ff(self.norm3(target))
        return target


class EncodecLatentModel(BaseLiveifyModel):
    def __init__(
        self,
        latent_dim: int = 128,
        context_length: int = 4,
        forward_context_length: int = 0,
        d_model: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
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

        self.input_proj = nn.Linear(latent_dim, d_model)
        self.context_proj = nn.Linear(latent_dim, d_model)

        self.layers = nn.ModuleList(
            [
                CrossAttentionBlock(d_model, num_heads, dropout)
                for _ in range(num_layers)
            ]
        )

        self.output_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, latent_dim),
        )

        nn.init.zeros_(self.output_proj[-1].weight)
        nn.init.zeros_(self.output_proj[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, C, T = x.shape
        assert S == self.num_slots

        ctx = x[:, :-1]  # (B, S-1, C, T)
        tgt = x[:, -1]  # (B, C, T)

        tgt_raw = rearrange(tgt, "b c t -> b t c")  # (B, T, latent_dim)
        ctx = rearrange(ctx, "b s c t -> b (s t) c")

        tgt_proj = self.input_proj(tgt_raw)  # (B, T, d_model)
        ctx_proj = self.context_proj(ctx)  # (B, (S-1)*T, d_model)

        for layer in self.layers:
            tgt_proj = layer(tgt_proj, ctx_proj)

        # output is a RESIDUAL.
        delta = self.output_proj(tgt_proj)  # (B, T, latent_dim)
        out = tgt_raw + delta  # (B, T, latent_dim)

        return out.transpose(1, 2)  # (B, latent_dim, T)


class EncodecLatentLightningModule(pl.LightningModule):
    """Train on Encodec continuous latents (no mel spectrograms)."""

    def __init__(
        self,
        model: "EncodecLatentModel",
        learning_rate: float = 1e-4,
        sample_rate: int = 48000,
        encodec_bandwidth: float = 6.0,
        encodec_sample_rate: int = 24000,
        cache_dir: str = "logs/encodec_latents",
        latent_noise_std: float = 0.3,
        forward_context_length: int = 0,
    ):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.data_sample_rate = sample_rate
        self.encodec_bandwidth = encodec_bandwidth
        self.encodec_sample_rate = encodec_sample_rate
        self.latent_noise_std = latent_noise_std
        self.forward_context_length = forward_context_length

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
        self.loss_l2 = nn.MSELoss()

        self.save_hyperparameters(ignore=["model", "encodec"])

    def _encode_audio(
        self, audio: torch.Tensor, cache_keys: Optional[List[str]] = None
    ) -> torch.Tensor:
        B, S, L = audio.shape

        cache_paths: List[Optional[Path]] = [None] * B
        latents: List[Optional[torch.Tensor]] = [None] * B
        missing_indices: List[int] = []

        if cache_keys is not None:
            for i, key in enumerate(cache_keys):
                hex_name = hashlib.sha1(key.encode("utf-8")).hexdigest()
                path = self.cache_dir / f"{hex_name}.pt"
                cache_paths[i] = path
                if path.exists():
                    latents[i] = torch.load(path, map_location=self.device)
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
                        f"[warning] Resampling from {self.data_sample_rate} Hz to {self.encodec_sample_rate} Hz inside Encodec encoder;"
                        " set dataset sr to encodec_sample_rate to avoid this cost."
                    )
                wav = torchaudio.functional.resample(
                    wav, self.data_sample_rate, self.encodec_sample_rate
                )
            wav = wav.to(self.device)

            # norm audio to [-1, 1] range before encoding
            wav_max = wav.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
            wav = wav / wav_max

            with torch.no_grad():
                lat_batch = self.encodec.encoder(wav)

            lat_batch = lat_batch.view(len(missing_indices), S, *lat_batch.shape[1:])

            for idx, sample_idx in enumerate(missing_indices):
                latents[sample_idx] = lat_batch[idx]
                if cache_paths[sample_idx] is not None:
                    torch.save(lat_batch[idx].cpu(), cache_paths[sample_idx])

        latents_tensor = torch.stack([t for t in latents if t is not None], dim=0)
        return latents_tensor.to(self.device)

    def forward(self, x):
        return self.model(x)

    def _align_time(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pt = pred.shape[-1]
        tt = target.shape[-1]
        if pt == tt:
            return pred, target
        if pt > tt:
            pred = pred[:, :, :tt]
        else:
            target = target[:, :, :pt]
        return pred, target

    def compute_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred, target = self._align_time(pred, target)

        channel_std = target.std(dim=(0, 2), keepdim=True).clamp(min=1e-3)
        pred_n = pred / channel_std
        target_n = target / channel_std

        l1 = F.l1_loss(pred_n, target_n)

        cos = (
            1.0
            - F.cosine_similarity(
                pred.transpose(1, 2), target.transpose(1, 2), dim=-1
            ).mean()
        )

        pred_diff = pred_n[:, :, 1:] - pred_n[:, :, :-1]
        tgt_diff = target_n[:, :, 1:] - target_n[:, :, :-1]
        grad_loss = F.l1_loss(pred_diff, tgt_diff)

        pred_diff2 = pred_diff[:, :, 1:] - pred_diff[:, :, :-1]
        tgt_diff2 = tgt_diff[:, :, 1:] - tgt_diff[:, :, :-1]
        grad2_loss = F.l1_loss(pred_diff2, tgt_diff2)

        return l1 + 0.1 * cos + 0.5 * grad_loss + 0.25 * grad2_loss

    def training_step(self, batch, batch_idx):
        studio_audio = batch["studio_audio"]
        live_audio = batch["live_audio"]
        cache_keys = batch.get("cache_key", None)

        x_lat = self._encode_audio(
            studio_audio, [f"studio::{k}" for k in cache_keys] if cache_keys else None
        )
        y_lat = self._encode_audio(
            live_audio, [f"live::{k}" for k in cache_keys] if cache_keys else None
        )

        if self.training and self.latent_noise_std > 0:
            noise_scale = x_lat.std() * self.latent_noise_std
            x_lat = x_lat + torch.randn_like(x_lat) * noise_scale

        y_pred = self(x_lat)
        target = y_lat[:, -1]
        loss = self.compute_loss(y_pred, target)

        lr = self.optimizers().param_groups[0]["lr"]
        self.log("train/lr", lr, prog_bar=True)
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        studio_audio = batch["studio_audio"]
        live_audio = batch["live_audio"]
        cache_keys = batch.get("cache_key", None)

        x_lat = self._encode_audio(
            studio_audio, [f"studio::{k}" for k in cache_keys] if cache_keys else None
        )
        y_lat = self._encode_audio(
            live_audio, [f"live::{k}" for k in cache_keys] if cache_keys else None
        )

        y_pred = self(x_lat)
        target = y_lat[:, -1]
        loss = self.compute_loss(y_pred, target)

        studio_latent = x_lat[:, -1]
        live_latent = y_lat[:, -1]
        trivial_loss = self.compute_loss(studio_latent, live_latent)

        self.log("val/loss", loss, prog_bar=True)
        self.log("val/trivial_baseline", trivial_loss, prog_bar=False)

        if not self._val_preview_logged:
            self._val_preview_logged = True
            self._log_latent_preview(y_pred.detach(), target.detach())
        return loss

    def on_validation_epoch_start(self):
        self._val_preview_logged = False

    def _log_latent_preview(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        if pred.numel() == 0 or target.numel() == 0:
            return

        pred = pred[0].detach().cpu().float()
        target = target[0].detach().cpu().float()

        pred_trace = pred.mean(dim=0)
        target_trace = target.mean(dim=0)

        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(pred_trace.numpy(), label="pred mean", alpha=0.8)
        ax.plot(target_trace.numpy(), label="target mean", alpha=0.8)
        ax.set_title("Encodec latent mean trace (sample 0)")
        ax.set_xlabel("time")
        ax.set_ylabel("mean latent")
        ax.legend()
        fig.tight_layout()

        logger = self.logger.experiment if hasattr(self.logger, "experiment") else None
        if logger is None:
            plt.close(fig)
            return

        try:
            if hasattr(logger, "add_figure"):
                logger.add_figure(
                    "val/latent_trace", fig, global_step=self.current_epoch
                )
            elif hasattr(logger, "log"):
                logger.log({"val/latent_trace": fig})
        finally:
            plt.close(fig)

    def on_before_optimizer_step(self, optimizer):
        if self.global_step % 10 == 0:
            trainable_params = [p for p in self.parameters() if p.requires_grad]
            total_norm = torch.nn.utils.clip_grad_norm_(
                trainable_params,
                max_norm=float("inf"),
                norm_type=2,
            )
            self.log("train/grad_norm", total_norm, on_step=True, on_epoch=False)

    def configure_optimizers(self):
        trainable_params = [p for p in self.parameters() if p.requires_grad]

        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.learning_rate,
            betas=(0.9, 0.98),
            weight_decay=0.05,  # increased from 1e-3
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=50,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "monitor": "val/loss",
            },
        }


def main() -> None:
    from torchinfo import summary

    model = EncodecLatentModel(
        latent_dim=128,
        context_length=4,
        num_layers=4,
        dropout=0.1,
    )

    summary(model, input_size=(1, 5, 128, 64))  # (batch, num_slots, latent_dim, time)


if __name__ == "__main__":
    main()
