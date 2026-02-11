import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from timm.models.layers import to_2tuple


# 3167087.62


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
    2D positional encoding for spectrogram patches
    """

    def __init__(
        self,
        embed_dim: int,
        num_patches_freq: int,
        num_patches_time: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_patches_freq = num_patches_freq
        self.num_patches_time = num_patches_time
        self.dropout = nn.Dropout(dropout)

        self.freq_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches_freq, embed_dim // 2)
        )
        self.time_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches_time, embed_dim // 2)
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.freq_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.time_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, num_patches, embed_dim)
        Returns:
            (batch, num_patches + 1, embed_dim)
        """
        B = x.shape[0]

        freq_pos = self.freq_pos_embed.repeat(1, self.num_patches_time, 1)
        time_pos = self.time_pos_embed.repeat_interleave(self.num_patches_freq, dim=1)
        pos_embed = torch.cat(
            [freq_pos, time_pos], dim=-1
        )  # (1, num_patches, embed_dim)

        x = x + pos_embed

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        return self.dropout(x)


class GRUContextModule(nn.Module):
    """
    Bidirectional GRU for capturing temporal context across patches
    """

    def __init__(self, embed_dim: int = 768, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()

        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=embed_dim // 2,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, embed_dim)
        Returns:
            (batch, seq_len, embed_dim)
        """
        gru_out, _ = self.gru(x)  # (B, seq_len, embed_dim)

        x = x + self.dropout(gru_out)
        x = self.norm(x)

        return x


class PatchReconstruction(nn.Module):
    """
    Reconstructs spectrogram from patch embeddings
    """

    def __init__(
        self,
        embed_dim: int = 768,
        patch_size: Tuple[int, int] = (16, 16),
        num_patches_freq: int = 8,
        num_patches_time: int = 64,
        out_channels: int = 1,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches_freq = num_patches_freq
        self.num_patches_time = num_patches_time

        patch_dim = patch_size[0] * patch_size[1] * out_channels
        self.to_patch = nn.Linear(embed_dim, patch_dim)

        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, num_patches, embed_dim)
        Returns:
            (batch, channels, freq, time)
        """
        B = x.shape[0]

        x = self.to_patch(x)  # (B, num_patches, patch_dim)

        x = x.view(
            B,
            self.num_patches_freq,
            self.num_patches_time,
            1,  # channels
            self.patch_size[0],
            self.patch_size[1],
        )

        x = x.permute(0, 3, 1, 4, 2, 5)  # (B, C, num_f, patch_f, num_t, patch_t)
        x = x.contiguous().view(
            B,
            1,
            self.num_patches_freq * self.patch_size[0],
            self.num_patches_time * self.patch_size[1],
        )

        x = self.upsample(x)

        return x


class TransformerEncoderLayer(nn.Module):
    """
    Transformer encoder layer with multi-head self-attention and feedforward network
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

        # ===== Multi-Head Self-Attention =====
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape

        x_norm = self.norm1(x)
        qkv = (
            self.qkv(x_norm)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, num_heads, N, head_dim)

        attn_output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.attn_drop.p if self.training else 0.0,
            is_causal=False,
        )  # reduces vram like crazy

        attn_output = attn_output.transpose(1, 2).reshape(B, N, C)
        attn_output = self.proj(attn_output)
        attn_output = self.proj_drop(attn_output)

        x = x + attn_output

        x = x + self.mlp(self.norm2(x))

        return x


class LiveifyModel(torch.nn.Module):
    def __init__(
        self,
        input_fdim: int = 128,
        input_tdim: int = 1024,
        patch_size: Tuple[int, int] = (16, 16),
        embed_dim: int = 768,
        num_transformer_layers: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        gru_layers: int = 2,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        in_channels: int = 1,
        out_channels: int = 1,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.input_fdim = input_fdim
        self.input_tdim = input_tdim

        # ===== Patch Embedding =====
        # input: (batch, channels, freq, time)
        # output: (batch, num_patches, embed_dim)
        self.patch_embed = PatchEmbedding(
            img_size=(input_fdim, input_tdim),
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
        )

        # ===== Positional Encoding =====
        # input: (batch, num_patches, embed_dim)
        # output: (batch, num_patches + 1, embed_dim)
        self.pos_embed = PositionalEncoding(
            embed_dim=embed_dim,
            num_patches_freq=self.patch_embed.num_patches_freq,
            num_patches_time=self.patch_embed.num_patches_time,
            dropout=dropout,
        )

        num_layers_pre_gru = num_transformer_layers // 2
        num_layers_post_gru = num_transformer_layers - num_layers_pre_gru

        # ===== Transformer Encoder Layers (Pre-GRU) =====
        # input: (batch, num_patches + 1, embed_dim)
        # output: (batch, num_patches + 1, embed_dim)
        self.transformer_pre_gru = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                )
                for _ in range(num_layers_pre_gru)
            ]
        )

        # ===== GRU Context Module =====
        # input: (batch, seq_len, embed_dim)
        # output: (batch, seq_len, embed_dim)
        self.gru_context = GRUContextModule(
            embed_dim=embed_dim, num_layers=gru_layers, dropout=dropout
        )

        # ===== Transformer Encoder Layers (Post-GRU) =====
        # input: (batch, num_patches + 1, embed_dim)
        # output: (batch, num_patches + 1, embed_dim)
        self.transformer_post_gru = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                )
                for _ in range(num_layers_post_gru)
            ]
        )

        self.norm = nn.LayerNorm(embed_dim)

        # ===== Patch Reconstruction =====
        # input: (batch, num_patches + 1, embed_dim)
        # output: (batch, channels, freq, time)
        self.patch_recon = PatchReconstruction(
            embed_dim=embed_dim,
            patch_size=patch_size,
            num_patches_freq=self.patch_embed.num_patches_freq,
            num_patches_time=self.patch_embed.num_patches_time,
            out_channels=out_channels,
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input spectrogram (batch, channels, freq, time)
        Returns:
            Reconstructed spectrogram (batch, channels, freq, time)
        """
        x = self.patch_embed(x)

        x = self.pos_embed(x)

        for layer in self.transformer_pre_gru:
            x = layer(x)

        cls_token = x[:, 0:1, :]
        x_patches = x[:, 1:, :]

        x_patches = self.gru_context(x_patches)

        x = torch.cat([cls_token, x_patches], dim=1)

        for layer in self.transformer_post_gru:
            x = layer(x)

        x = self.norm(x)

        x = x[:, 1:, :]  # dont use cls token for reconstruction

        x = self.patch_recon(x)

        return x


def main() -> None:
    from torchinfo import summary

    import torch.profiler as profiler

    model = LiveifyModel(
        input_fdim=128,
        input_tdim=1024,
        patch_size=(16, 16),
        embed_dim=512,
        num_transformer_layers=3,
        num_heads=4,
        mlp_ratio=4.0,
        gru_layers=2,
        dropout=0.1,
        attention_dropout=0.1,
        in_channels=1,
        out_channels=1,
    ).to("cuda")

    summary(model, input_size=(1, 1, 128, 1024))

    dummy_input = torch.randn(1, 1, 128, 1024).to("cuda")

    with profiler.profile(
        activities=[
            profiler.ProfilerActivity.CPU,
            profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        with profiler.record_function("model_inference"):
            output = model(dummy_input)

    # prof.export_chrome_trace("liveify_profiler_trace.json")
    with open("profiler_stats.txt", "w") as f:
        f.write(
            prof.key_averages().table(
                sort_by="cuda_time_total",
                row_limit=20,
            )
        )
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))


if __name__ == "__main__":
    main()
