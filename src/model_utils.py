"""Utility components for Liveify models (patch embedding, positional encoding, etc.)."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


def to_2tuple(x):
    """Convert to 2-tuple if int, else return as is."""
    if isinstance(x, int):
        return (x, x)
    return x


class PatchEmbedding(nn.Module):
    """Convert spectrogram to patch embeddings."""

    def __init__(
        self, img_size=(128, 1024), patch_size=16, in_channels=1, embed_dim=768
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = to_2tuple(patch_size)
        self.num_patches_freq = img_size[0] // self.patch_size[0]
        self.num_patches_time = img_size[1] // self.patch_size[1]
        self.n_patches = self.num_patches_freq * self.num_patches_time

        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=self.patch_size, stride=self.patch_size
        )

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x


class PositionalEncoding(nn.Module):
    """
    2D positional encoding for spectrogram patches + segment positional encoding
    to distinguish which context slot each token came from.
    """

    def __init__(
        self,
        embed_dim: int,
        num_patches_freq: int,
        num_patches_time: int,
        num_slots: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_patches_freq = num_patches_freq
        self.num_patches_time = num_patches_time
        self.num_slots = num_slots
        self.dropout = nn.Dropout(dropout)

        # 2D spatial positional encoding (freq + time)
        self.freq_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches_freq, embed_dim // 2)
        )
        self.time_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches_time, embed_dim // 2)
        )

        # one vector per slot
        self.segment_pos_embed = nn.Parameter(torch.zeros(1, num_slots, 1, embed_dim))

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self._init_weights()

    def _init_weights(self):
        nn.init.zeros_(self.freq_pos_embed)
        nn.init.zeros_(self.time_pos_embed)
        nn.init.zeros_(self.segment_pos_embed)
        nn.init.zeros_(self.cls_token)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, num_slots, patches_per_slot, embed_dim)
        Returns:
            (batch, num_slots * patches_per_slot + 1, embed_dim)
               — cls token prepended
        """
        B, S, P, C = x.shape

        freq_pos = self.freq_pos_embed.repeat(1, self.num_patches_time, 1)
        time_pos = self.time_pos_embed.repeat_interleave(self.num_patches_freq, dim=1)
        spatial_pos = torch.cat([freq_pos, time_pos], dim=-1)  # (1, P, embed_dim)

        x = x + spatial_pos.unsqueeze(1)  # broadcast over slots: (B, S, P, C)

        x = x + self.segment_pos_embed  # (B, S, P, C)

        x = x.reshape(B, S * P, C)  # (B, S*P, C)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, S*P + 1, C)

        return self.dropout(x)


class PatchReconstruction(nn.Module):
    """
    Reconstructs a single spectrogram segment from the last slot's patch embeddings.
    Optionally upsamples using PixelShuffle for higher resolution output.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        patch_size: Tuple[int, int] = (16, 16),
        num_patches_freq: int = 8,
        num_patches_time: int = 64,
        out_channels: int = 1,
        upscale_factor: int = 1,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches_freq = num_patches_freq
        self.num_patches_time = num_patches_time
        self.upscale_factor = upscale_factor

        # to_patch outputs out_channels * upscale_factor^2 for PixelShuffle
        patch_dim = patch_size[0] * patch_size[1] * out_channels * (upscale_factor**2)
        self.to_patch = nn.Linear(embed_dim, patch_dim)

        self.pixel_shuffle = (
            nn.PixelShuffle(upscale_factor) if upscale_factor > 1 else None
        )

        self.refine = nn.Sequential(
            nn.Conv2d(out_channels, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, patches_per_slot, embed_dim) — last slot's tokens only
        Returns:
            (batch, out_channels, freq * upscale_factor, time * upscale_factor)
        """
        B = x.shape[0]

        x = self.to_patch(
            x
        )  # (B, P, patch_dim) where patch_dim = patch_h*patch_w*out_c*r^2

        x = x.view(
            B,
            self.num_patches_freq,
            self.num_patches_time,
            1 * (self.upscale_factor**2),  # out_channels * r^2
            self.patch_size[0],
            self.patch_size[1],
        )

        x = x.permute(0, 3, 1, 4, 2, 5)  # (B, C*r^2, num_f, patch_f, num_t, patch_t)
        x = x.contiguous().view(
            B,
            1 * (self.upscale_factor**2),
            self.num_patches_freq * self.patch_size[0],
            self.num_patches_time * self.patch_size[1],
        )

        if self.pixel_shuffle is not None:
            x = self.pixel_shuffle(x)  # (B, 1, freq*r, time*r)

        x = self.refine(x)

        return x


class TransformerEncoderLayer(nn.Module):
    """
    Transformer encoder layer with multi-head self-attention and feedforward network.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == embed_dim
        ), "embed_dim must be divisible by num_heads"

        self.norm1 = nn.LayerNorm(embed_dim)

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.attn_drop = nn.Dropout(attention_dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.mlp[-2].weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape

        x_norm = self.norm1(x)
        qkv = (
            self.qkv(x_norm)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn_output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.attn_drop.p if self.training else 0.0,
            is_causal=False,
        )

        attn_output = attn_output.transpose(1, 2).reshape(B, N, C)
        attn_output = self.proj(attn_output)
        attn_output = self.proj_drop(attn_output)

        x = x + attn_output
        x = x + self.mlp(self.norm2(x))

        return x
