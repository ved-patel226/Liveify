import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.tuner import Tuner

from dataset.dataset import StudioLiveDataModule


# optimal lr: 4.5e-064.5e-06
class LiveifyModel(torch.nn.Module):
    def __init__(
        self,
        input_sr=88200,
        output_sr=88200,
        hidden_channels=128,
        encoder_strides=[8, 8, 4, 4],
        transformer_dim=256,
        num_heads=8,
        num_layers=4,
        lr=1e-4,
    ):
        super(LiveifyModel, self).__init__()

        self.input_sr = input_sr
        self.output_sr = output_sr
        self.lr = lr
        self.encoder_strides = encoder_strides

        total_stride = 1
        for s in encoder_strides:
            total_stride *= s
        self.patch_size = total_stride
        self.num_patches = (input_sr + self.patch_size - 1) // self.patch_size

        # ===== Encoder  =====
        encoder_layers = []
        in_channels = 1

        num_stages = len(encoder_strides)
        channels = self._calculate_channel_progression(
            start_channels=hidden_channels // 2,
            end_channels=transformer_dim,
            num_stages=num_stages,
        )

        for i, stride in enumerate(encoder_strides):
            out_channels = channels[i]

            kernel_size = stride * 2 + 1
            padding = kernel_size // 2

            encoder_layers.extend(
                [
                    nn.Conv1d(
                        in_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                    ),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(),
                ]
            )
            in_channels = out_channels

        self.patch_embed = nn.Sequential(*encoder_layers)
        self.encoder_channels = channels

        self.pos_embed = nn.Parameter(
            torch.randn(1, transformer_dim, self.num_patches) * 0.02
        )

        # ===== Transformer =====
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim,
            nhead=num_heads,
            dim_feedforward=transformer_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # ===== Decoder  =====
        decoder_layers = []
        decoder_strides = encoder_strides[::-1]

        decoder_channels = [transformer_dim] + channels[::-1][1:] + [1]

        in_channels = transformer_dim
        for i, stride in enumerate(decoder_strides):
            out_channels = decoder_channels[i + 1]

            kernel_size = stride * 2
            padding = kernel_size // 2 - stride // 2

            decoder_layers.append(
                nn.ConvTranspose1d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                )
            )

            if i < len(decoder_strides) - 1:
                decoder_layers.extend(
                    [
                        nn.BatchNorm1d(out_channels),
                        nn.ReLU(),
                    ]
                )
            else:
                decoder_layers.append(nn.Tanh())

            in_channels = out_channels

        self.patch_decode = nn.Sequential(*decoder_layers)

        self._init_near_zero()

    def _calculate_channel_progression(self, start_channels, end_channels, num_stages):
        """
        Calculate smooth channel progression from start to end.
        Uses geometric progression for smooth exponential growth.
        """
        if num_stages == 1:
            return [end_channels]

        ratio = (end_channels / start_channels) ** (1 / (num_stages - 1))

        channels = []
        for i in range(num_stages):
            ch = int(start_channels * (ratio**i))
            # round to nearest 8
            ch = ((ch + 7) // 8) * 8
            channels.append(ch)

        channels[-1] = end_channels

        return channels

    def _init_near_zero(self):
        """Initialize weights near zero for residual learning."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
                nn.init.normal_(m.weight, mean=0.0, std=1e-4)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        if hasattr(self, "patch_decode"):
            for layer in reversed(list(self.patch_decode.children())):
                if isinstance(layer, nn.ConvTranspose1d):
                    nn.init.normal_(layer.weight, mean=0.0, std=1e-4)
                    break

    def forward(self, x):
        """
        Args:
            x: (batch_size, sequence_length) raw audio
        Returns:
            (batch_size, sequence_length) processed audio
        """
        _, seq_len = x.shape

        pad_len = (self.patch_size - seq_len % self.patch_size) % self.patch_size
        if pad_len > 0:
            x = F.pad(x, (0, pad_len))

        x_in = x.unsqueeze(1)

        # ===== ENCODE =====
        patches = self.patch_embed(x_in)
        patches = patches + self.pos_embed

        # ===== TRANSFORM =====
        patches = patches.permute(0, 2, 1)
        transformed = self.transformer(patches)
        transformed = transformed.permute(0, 2, 1)

        # ===== DECODE =====
        residual = self.patch_decode(transformed)

        if pad_len > 0:
            residual = residual[:, :, :-pad_len]

        residual = residual.squeeze(1)

        return x[:, :seq_len] + residual


class LiveifyLRFinderWrapper(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.lr = model.lr

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = torch.nn.functional.l1_loss(y_pred, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = torch.nn.functional.l1_loss(y_pred, y)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


def find_lr(model):
    batch_size = 4
    seq_length = 22050 * 5  # 5 seconds
    device = next(model.parameters()).device
    x = torch.randn(batch_size, seq_length).to(device)

    print(f"Input shape: {x.shape}")
    y = model(x)
    print(f"Output shape: {y.shape}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    print("\n" + "=" * 50)
    print("Testing Learning Rate Finder...")
    print("=" * 50 + "\n")

    datamodule = StudioLiveDataModule(
        studio_dir="./dataset/studio",
        live_dir="./dataset/live",
        batch_size=4,
        segment_duration=5.0,
        development_mode=False,
        num_workers=2,
    )

    lightning_model = LiveifyLRFinderWrapper(model)

    trainer = pl.Trainer(
        max_epochs=5,
        accelerator="auto",
        devices=1,
        enable_checkpointing=False,
        logger=False,
    )

    try:
        tuner = Tuner(trainer)
        lr_finder = tuner.lr_find(
            lightning_model,
            datamodule,
            min_lr=1e-6,
            max_lr=1e-1,
            num_training=100,
            mode="exponential",
        )

        fig = lr_finder.plot(suggest=True)
        fig.savefig("lr_finder_test.png")
        print(f"Learning rate finder plot saved to: lr_finder_test.png")

        suggested_lr = lr_finder.suggestion()
        print(f"Suggested learning rate: {suggested_lr:.2e}")

    except Exception as e:
        print(f"Error running lr_find: {e}")
        print("Make sure dataset paths are correct and data is available.")


if __name__ == "__main__":
    from torchinfo import summary

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LiveifyModel(
        input_sr=110250,
        output_sr=110250,
        hidden_channels=128,
        encoder_strides=[8, 4, 4],
        transformer_dim=(512 + 256) // 2,
        num_heads=8,
        num_layers=4,
        lr=1e-4,
    ).to(device)

    find_lr(model)

    # summary(
    #     model,
    #     input_size=(4, 110250),
    #     col_names=["input_size", "output_size", "num_params"],
    #     depth=3,
    #     device=device,
    #     verbose=1,
    # )
