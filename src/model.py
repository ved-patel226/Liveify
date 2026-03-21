import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from timm.models.layers import to_2tuple


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

        # 2D spatial positional encoding (freq + time), split embed_dim in half
        self.freq_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches_freq, embed_dim // 2)
        )
        self.time_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches_time, embed_dim // 2)
        )

        # Segment positional encoding: one vector per slot
        # Added to every patch token belonging to that slot
        self.segment_pos_embed = nn.Parameter(torch.zeros(1, num_slots, 1, embed_dim))

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self._init_weights()

    def _init_weights(self):
        # Zero-init so positional encodings don't perturb tokens at init time,
        # which would break the encoder->decoder identity path in LiveifyModel.
        # They are fully learned and diverge from zero quickly during training.
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

        # 2D spatial positional encoding — same for every slot
        freq_pos = self.freq_pos_embed.repeat(1, self.num_patches_time, 1)
        time_pos = self.time_pos_embed.repeat_interleave(self.num_patches_freq, dim=1)
        spatial_pos = torch.cat([freq_pos, time_pos], dim=-1)  # (1, P, embed_dim)

        x = x + spatial_pos.unsqueeze(1)  # broadcast over slots: (B, S, P, C)

        # Segment positional encoding — unique per slot, broadcast over patches
        x = x + self.segment_pos_embed  # (B, S, P, C)

        # Flatten slots and patches into a single sequence
        x = x.reshape(B, S * P, C)  # (B, S*P, C)

        # Prepend cls token
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

        # PixelShuffle for upsampling if upscale_factor > 1
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

        # Apply PixelShuffle if upscale_factor > 1
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


class LiveifyModel(torch.nn.Module):
    """
    Processes context_length+1 spectrogram segments:
      - Slots [0 .. context_length-1] are context (past studio segments).
      - Slot [context_length] is the current target segment to reconstruct.

    All slots are patch-embedded independently with shared weights, then
    concatenated into a single token sequence for the transformer.
    Only the last slot's tokens are decoded back into a spectrogram.
    """

    def __init__(
        self,
        input_fdim: int = 128,
        input_tdim: int = 1024,
        patch_size: Tuple[int, int] = (16, 16),
        embed_dim: int = 768,
        num_transformer_layers: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        in_channels: int = 1,
        out_channels: int = 1,
        context_length: int = 4,  # number of past context slots (not counting current)
        upscale_factor: int = 1,  # PixelShuffle upscale factor for super-resolution
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.input_fdim = input_fdim
        self.input_tdim = input_tdim
        self.context_length = context_length
        self.num_slots = context_length + 1  # context + current
        self.upscale_factor = upscale_factor

        patch_size_tuple = (
            to_2tuple(patch_size) if isinstance(patch_size, int) else patch_size
        )

        # ===== Patch Embedding (shared across all slots) =====
        # input:  (batch, channels, freq, time)   — one slot at a time
        # output: (batch, patches_per_slot, embed_dim)
        self.patch_embed = PatchEmbedding(
            img_size=(input_fdim, input_tdim),
            patch_size=patch_size_tuple[0],
            in_channels=in_channels,
            embed_dim=embed_dim,
        )

        self.patches_per_slot = self.patch_embed.n_patches

        # ===== Positional Encoding =====
        # input:  (batch, num_slots, patches_per_slot, embed_dim)
        # output: (batch, num_slots * patches_per_slot + 1, embed_dim)
        self.pos_embed = PositionalEncoding(
            embed_dim=embed_dim,
            num_patches_freq=self.patch_embed.num_patches_freq,
            num_patches_time=self.patch_embed.num_patches_time,
            num_slots=self.num_slots,
            dropout=dropout,
        )

        # ===== Transformer Encoder =====
        # attends over all slots simultaneously
        self.transformer_layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                )
                for _ in range(num_transformer_layers)
            ]
        )

        self.norm = nn.LayerNorm(embed_dim)

        # ===== Patch Reconstruction (last slot only) =====
        # input:  (batch, patches_per_slot, embed_dim)
        # output: (batch, out_channels, freq*upscale_factor, time*upscale_factor)
        self.patch_recon = PatchReconstruction(
            embed_dim=embed_dim,
            patch_size=patch_size_tuple,
            num_patches_freq=self.patch_embed.num_patches_freq,
            num_patches_time=self.patch_embed.num_patches_time,
            out_channels=out_channels,
            upscale_factor=upscale_factor,
        )

    #     self.apply(self._init_weights)
    #     # Calibration pass: fit to_patch so output ≈ input at init.
    #     # Must run after apply() so all other weights are already set.
    #     self._init_decoder_to_identity()

    # def _init_weights(self, m):
    #     if isinstance(m, nn.Linear):
    #         if m is self.patch_recon.to_patch:
    #             pass
    #         else:
    #             nn.init.trunc_normal_(m.weight, std=0.02)
    #             if m.bias is not None:
    #                 nn.init.constant_(m.bias, 0)

    # def _init_decoder_to_identity(self) -> None:
    #     device = self.patch_embed.proj.weight.device
    #     dtype = self.patch_embed.proj.weight.dtype

    #     patch_h, patch_w = self.patch_embed.patch_size
    #     in_ch = self.patch_embed.proj.in_channels
    #     embed_dim = self.patch_embed.proj.out_channels
    #     patch_dim = in_ch * patch_h * patch_w

    #     n_cal = max(4 * embed_dim, 1024)

    #     with torch.no_grad():
    #         cal_patches = torch.randn(
    #             n_cal, in_ch, patch_h, patch_w, device=device, dtype=torch.float32
    #         )

    #         tokens = self.patch_embed.proj(cal_patches)  # (n_cal, embed_dim, 1, 1)
    #         tokens = tokens.view(n_cal, embed_dim)  # (n_cal, embed_dim)
    #         tokens_normed = self.norm(tokens)  # (n_cal, embed_dim)

    #         # Least-squares: find W, b  s.t.  tokens_normed @ W^T + b ≈ patches_flat
    #         patches_flat = cal_patches.view(n_cal, patch_dim)  # (n_cal, patch_dim)

    #         A = torch.cat(
    #             [
    #                 tokens_normed,
    #                 torch.ones(n_cal, 1, device=device, dtype=torch.float32),
    #             ],
    #             dim=1,
    #         )  # (n_cal, embed_dim+1)

    #         # lstsq solution X: (embed_dim+1, patch_dim)
    #         X = torch.linalg.lstsq(A, patches_flat, driver="gelsd").solution

    #         W_fit = X[:embed_dim].T  # (patch_dim, embed_dim)
    #         b_fit = X[embed_dim]  # (patch_dim,)

    #         self.patch_recon.to_patch.weight.copy_(W_fit.to(dtype))
    #         if self.patch_recon.to_patch.bias is not None:
    #             self.patch_recon.to_patch.bias.copy_(b_fit.to(dtype))

    #     last_conv = self.patch_recon.refine[-1]
    #     if isinstance(last_conv, nn.Conv2d):
    #         nn.init.zeros_(last_conv.weight)
    #         if last_conv.bias is not None:
    #             nn.init.zeros_(last_conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, num_slots, channels, freq, time)
               num_slots = context_length + 1
               last slot [:, -1] is the current segment to reconstruct.
        Returns:
            (batch, out_channels, freq*upscale_factor, time*upscale_factor)  — upsampled reconstruction
        """
        B, S, C, freq_dim, time_dim = x.shape
        assert (
            S == self.num_slots
        ), f"Expected {self.num_slots} slots (context_length={self.context_length} + 1), got {S}"

        # CRITICAL: Save input for residual connection AND upsample if needed
        # The decoder may output higher resolution via PixelShuffle upscaling
        target_input = x[:, -1, :, :, :]  # (B, C, freq_dim, time_dim)
        if self.upscale_factor > 1:
            # Upsample target to match output resolution for loss computation
            target_input = F.interpolate(
                target_input,
                scale_factor=self.upscale_factor,
                mode="nearest",
            )  # (B, C, freq_dim*r, time_dim*r)

        x_flat = x.view(B * S, C, freq_dim, time_dim)  # (B*S, C, freq_dim, time_dim)
        tokens = self.patch_embed(x_flat)  # (B*S, P, embed_dim)
        tokens = tokens.view(B, S, self.patches_per_slot, -1)  # (B, S, P, embed_dim)

        tokens = self.pos_embed(tokens)  # (B, S*P + 1, embed_dim)

        for layer in self.transformer_layers:
            tokens = layer(tokens)

        tokens = self.norm(tokens)

        tokens = tokens[:, 1:, :]  # (B, S*P, embed_dim)
        tokens = tokens.view(B, S, self.patches_per_slot, -1)  # (B, S, P, embed_dim)
        last_slot_tokens = tokens[:, -1, :, :]  # (B, P, embed_dim)

        delta = self.patch_recon(last_slot_tokens)  # (B, out_channels, F, T)

        # ADD RESIDUAL CONNECTION - model predicts delta from input
        out = target_input + delta

        return out


def main() -> None:
    from torchinfo import summary

    upscale = 2  # 2x super-resolution via PixelShuffle
    model = LiveifyModel(
        input_fdim=256,
        input_tdim=64,
        patch_size=(16, 16),
        embed_dim=256,
        num_transformer_layers=2,
        num_heads=2,
        mlp_ratio=4.0,
        dropout=0.1,
        attention_dropout=0.1,
        in_channels=1,
        out_channels=1,
        context_length=4,
        upscale_factor=upscale,
    ).to("cuda")

    dummy_input = torch.randn(2, 5, 1, 256, 64).to("cuda")
    summary(model, input_data=dummy_input)

    out = model(dummy_input)
    print(f"Input:  {dummy_input.shape}")
    print(f"Output: {out.shape}")  # (2, 1, 512, 128)


if __name__ == "__main__":
    main()
