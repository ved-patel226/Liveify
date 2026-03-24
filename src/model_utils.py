"""Utility components for Liveify models."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleSTFTLoss(nn.Module):
    def __init__(self, fft_sizes=(512, 1024, 2048), hop_divisor=4):
        super().__init__()
        self.fft_sizes = fft_sizes
        self.hop_divisor = hop_divisor

    def forward(self, pred_audio, target_audio):
        loss = 0.0
        for n_fft in self.fft_sizes:
            hop = n_fft // self.hop_divisor
            window = torch.hann_window(n_fft, device=pred_audio.device)
            pred_stft = torch.stft(
                pred_audio.squeeze(1), n_fft, hop, window=window, return_complex=True
            )
            tgt_stft = torch.stft(
                target_audio.squeeze(1), n_fft, hop, window=window, return_complex=True
            )
            pred_mag, tgt_mag = pred_stft.abs(), tgt_stft.abs()
            loss += F.l1_loss(pred_mag, tgt_mag)
            loss += F.l1_loss(torch.log(pred_mag + 1e-7), torch.log(tgt_mag + 1e-7))
        return loss / len(self.fft_sizes)


class LearnedPositionalEncoding1D(nn.Module):
    def __init__(self, embed_dim, max_len=4096, dropout=0.1):
        super().__init__()
        self.pos = nn.Parameter(torch.zeros(1, max_len, embed_dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(x + self.pos[:, : x.shape[1]])


class CrossAttentionBlock(nn.Module):
    """Cross-attention block with PROPER regularization.

    Changes from original:
      - Dropout after BOTH attention outputs (was missing entirely)
      - Dropout inside FFN (was missing entirely)
      - Configurable FF expansion (was hardcoded 4×)
      - Stochastic depth / drop-path
    """

    def __init__(self, d_model, nhead, ff_mult=2, dropout=0.3, drop_path=0.0):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.cross_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ff_mult, d_model),
            nn.Dropout(dropout),
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.attn_drop1 = nn.Dropout(dropout)
        self.attn_drop2 = nn.Dropout(dropout)
        self.drop_path_rate = drop_path

    def _drop_path(self, x):
        """Stochastic depth: randomly skip this residual branch."""
        if not self.training or self.drop_path_rate == 0.0:
            return x
        keep = 1.0 - self.drop_path_rate
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.bernoulli(torch.full(shape, keep, device=x.device))
        return x * mask / keep

    def forward(self, target, context):
        h = self.norm1(target)
        target = target + self._drop_path(self.attn_drop1(self.self_attn(h, h, h)[0]))

        h = self.norm2(target)
        target = target + self._drop_path(
            self.attn_drop2(self.cross_attn(h, context, context)[0])
        )

        target = target + self._drop_path(self.ff(self.norm3(target)))
        return target
