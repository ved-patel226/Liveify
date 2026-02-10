import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.tuner import Tuner

from dataset.dataset import StudioLiveDataModule


#! Watch this performance, HIGHLY EXPIREMENTAL
# * Grows logarithmically in complexity with input, but it might not be as good as LSTM for shorter contexts
class HierarchicalContext(nn.Module):
    def __init__(self, dim=512, num_levels=3):
        super().__init__()
        self.num_levels = num_levels

        self.downsamples = nn.ModuleList()

        self.shared_process = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=8,
                dim_feedforward=dim * 2,
                dropout=0.1,
                batch_first=True,
                norm_first=True,
            ),
            num_layers=2,
        )

        self.weights = nn.ParameterList()

        for i in range(num_levels):
            if i > 0:
                self.downsamples.append(
                    nn.Conv1d(dim, dim, kernel_size=4, stride=4, padding=0)
                )
            else:
                self.downsamples.append(nn.Identity())

            self.weights.append(nn.Parameter(torch.tensor(1.0 / (i + 1))))

    def forward(self, x):
        B, C, D = x.shape
        outputs = []
        current = x.transpose(1, 2)

        for i in range(self.num_levels):
            downsampled = self.downsamples[i](current)
            downsampled = downsampled.transpose(1, 2)

            # ⭐ Use shared transformer
            processed = self.shared_process(downsampled)

            processed = processed.transpose(1, 2)
            if processed.shape[2] != C:
                processed = F.interpolate(
                    processed, size=C, mode="linear", align_corners=False
                )
            processed = processed.transpose(1, 2)

            outputs.append(self.weights[i] * processed)

        return sum(outputs)


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
        context_length=1,
        lr=1e-4,
    ):
        super(LiveifyModel, self).__init__()

        self.input_sr = input_sr
        self.output_sr = output_sr
        self.lr = lr
        self.encoder_strides = encoder_strides
        self.context_length = context_length

        total_stride = 1
        for s in encoder_strides:
            total_stride *= s
        self.patch_size = total_stride
        self.num_patches = (input_sr + self.patch_size - 1) // self.patch_size

        # ===== Encoder =====
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
                    nn.GELU(),
                ]
            )
            in_channels = out_channels

        self.patch_embed = nn.Sequential(*encoder_layers)
        self.encoder_channels = channels

        self.pos_embed = nn.Parameter(
            torch.randn(1, transformer_dim, self.num_patches) * 0.001
        )

        # # ===== LSTM =====
        # # TODO: check out LSTMx for longer sequences, mabye add an option for that https://arxiv.org/abs/2211.13227
        # self.lstm = nn.LSTM(
        #     input_size=transformer_dim,
        #     hidden_size=lstm_hidden,
        #     num_layers=lstm_layers,
        #     batch_first=True,
        #     bidirectional=True,
        #     dropout=0.2 if lstm_layers > 1 else 0.0,
        # )
        # self.lstm_proj = nn.Linear(lstm_hidden * 2, transformer_dim)
        # self.lstm_norm = nn.LayerNorm(transformer_dim)

        self.hierarchical_context = HierarchicalContext(
            dim=transformer_dim, num_levels=3
        )
        # ===== Transformer for global attention =====
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

        # ===== Decoder =====
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
                # GroupNorm instead of BatchNorm cuz it was pretty inconsistant output across segments
                num_groups = 8
                while out_channels % num_groups != 0 and num_groups > 1:
                    num_groups //= 2
                decoder_layers.extend(
                    [
                        nn.GroupNorm(num_groups, out_channels),
                        nn.ELU(alpha=1.0),  # ELU preserves negative values
                    ]
                )

            in_channels = out_channels

        self.decoder = nn.Sequential(*decoder_layers)

        self.skip_convs = nn.ModuleList()
        for i in range(len(decoder_strides) - 1):
            enc_ch = channels[::-1][i]
            dec_ch = decoder_channels[i + 1]
            self.skip_convs.append(nn.Conv1d(enc_ch, dec_ch, 1))

        self._init_weights()

    def _calculate_channel_progression(self, start_channels, end_channels, num_stages):
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

    def _init_weights(self):
        """Weight initialization."""
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        decoder_convs = [
            m for m in self.decoder.modules() if isinstance(m, nn.ConvTranspose1d)
        ]
        if decoder_convs:
            nn.init.zeros_(decoder_convs[-1].weight)
            if decoder_convs[-1].bias is not None:
                nn.init.zeros_(decoder_convs[-1].bias)

    def forward(self, x):
        """
        Args:
            x: (batch_size, context_length, segment_length) if context_length > 1
               or (batch_size, segment_length) if context_length == 1
        Returns:
            (batch_size, context_length, segment_length) if context_length > 1
            or (batch_size, segment_length) if context_length == 1
        """
        has_context = x.dim() == 3
        if has_context:
            B, C, S = x.shape  # batch, context_length, segment_length
        else:
            B, S = x.shape
            C = 1
            x = x.unsqueeze(1)  # (B, 1, S)

        seq_len = S

        # Save original input for residual connection BEFORE padding
        x_orig = x  # (B, C, S)

        pad_len = (self.patch_size - seq_len % self.patch_size) % self.patch_size
        if pad_len > 0:
            x = F.pad(x, (0, pad_len))
            S_padded = S + pad_len
        else:
            S_padded = S

        # Reshape to process all frames through encoder: (B*C, 1, S_padded)
        x_flat = x.reshape(B * C, 1, S_padded)

        # ===== ENCODE =====
        encoder_features = []
        out = x_flat
        for layer in self.patch_embed:
            out = layer(out)
            if isinstance(layer, nn.GELU):
                encoder_features.append(out)

        patches = encoder_features[-1]  # (B*C, transformer_dim, num_patches)
        patches = patches + self.pos_embed

        # # ===== LSTM =====
        # # patches: (B*C, D, T) -> (B*C, T, D) for LSTM
        # lstm_in = patches.permute(0, 2, 1)
        # lstm_out, _ = self.lstm(lstm_in)  # (B*C, T, lstm_hidden*2)
        # lstm_out = self.lstm_proj(lstm_out)  # (B*C, T, transformer_dim)
        # lstm_out = self.lstm_norm(lstm_out + lstm_in)  # residual connection

        lstm_in = patches.permute(0, 2, 1)
        lstm_out = lstm_in

        if C > 1:
            T = lstm_out.shape[1]
            D = lstm_out.shape[2]
            context_features = lstm_out.reshape(B, C, T, D)

            frame_summary = context_features.mean(dim=2)  # (B, C, D)

            frame_ctx = self.hierarchical_context(frame_summary)  # (B, C, D)

            frame_ctx_expanded = frame_ctx.unsqueeze(2).expand_as(context_features)
            context_features = context_features + 0.1 * frame_ctx_expanded

            lstm_out = context_features.reshape(B * C, T, D)

        # ===== TRANSFORMER: global attention =====
        transformed = self.transformer(lstm_out)  # (B*C, T, D)
        transformed = transformed.permute(0, 2, 1)  # (B*C, D, T)

        # ===== DECODE with skip connections =====
        modules = list(self.decoder)
        idx = 0
        decoder_out = transformed
        num_stages = len([m for m in modules if isinstance(m, nn.ConvTranspose1d)])

        for i in range(num_stages):
            conv = modules[idx]
            assert isinstance(conv, nn.ConvTranspose1d)
            decoder_out = conv(decoder_out)
            idx += 1

            if i < len(self.skip_convs) and i < len(encoder_features):
                encoder_feat = encoder_features[-(i + 1)]

                if decoder_out.shape[2] != encoder_feat.shape[2]:
                    encoder_feat = F.interpolate(
                        encoder_feat,
                        size=decoder_out.shape[2],
                        mode="linear",
                        align_corners=False,
                    )

                skip_feat = self.skip_convs[i](encoder_feat)
                decoder_out = decoder_out + skip_feat

            if i < num_stages - 1:
                norm = modules[idx]
                act = modules[idx + 1]
                decoder_out = norm(decoder_out)
                decoder_out = act(decoder_out)
                idx += 2

        output = decoder_out.squeeze(1)  # (B*C, S_padded)
        output = output[:, :seq_len]  # trim padding

        input_flat = x_orig.reshape(B * C, seq_len)
        output = input_flat + output  # residual learning

        if has_context:
            output = output.reshape(B, C, seq_len)
        else:
            output = output.reshape(B, seq_len)

        return output


class LiveifyLRFinderWrapper(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.lr = model.lr
        self.context_length = model.context_length

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
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=15, min_lr=1e-8, verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }


def find_lr(model):
    batch_size = 4
    seq_length = int(22050 * 0.02)  # 20ms = 441 samples
    context_length = model.context_length
    device = next(model.parameters()).device

    if context_length > 1:
        x = torch.randn(batch_size, context_length, seq_length).to(device)
    else:
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
        segment_duration=0.02,
        context_length=context_length,
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
            min_lr=1e-15,
            max_lr=1e-1,
            num_training=250,
            mode="exponential",
            early_stop_threshold=15,
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
    input_length = int(22050 * 0.5)  # 0.5s = 11025 samples
    context_length = 64
    model = LiveifyModel(
        input_sr=input_length,
        output_sr=input_length,
        hidden_channels=256,
        encoder_strides=[8, 4, 4],
        transformer_dim=256,
        num_heads=8,
        num_layers=6,
        context_length=context_length,
        lr=1e-5,
    ).to(device)

    # find_lr(model)

    print(f"Model Summary:")
    print(f"  Input: (batch, {context_length}, {input_length})")
    print(f"  Output: (batch, {context_length}, {input_length})")
    print()

    summary(
        model,
        input_size=(4, context_length, input_length),
        col_names=["input_size", "output_size", "num_params"],
        depth=3,
        device=device,
        verbose=1,
    )
