import torch

torch.set_float32_matmul_precision("high")

from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger


import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pathlib import Path
import argparse

from model import LiveifyModel
from dataset_utils.dataset import StudioLiveDataModule


class SpectrogramLoss(nn.Module):
    """Loss for spectrogram-based models."""

    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()

    def forward(self, pred, target):
        """Compute combined MSE and L1 loss on spectrograms.

        Args:
            pred: (batch, channels, freq, time) or (batch, context, channels, freq, time)
            target: same shape as pred
        """
        mse = self.mse_loss(pred, target)

        l1 = self.l1_loss(pred, target)

        return 0.5 * mse + 0.5 * l1


class LiveifyLightningModule(pl.LightningModule):
    def __init__(
        self,
        model: LiveifyModel,
        learning_rate: float = 1e-4,
        sample_rate: int = 22050,
    ):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.sample_rate = sample_rate

        self.loss_fn = SpectrogramLoss()

        self.validation_outputs = []

        self.save_hyperparameters(ignore=["model"])

    def forward(self, x):
        target_f = self.model.input_fdim
        target_t = self.model.input_tdim

        if x.shape[2] < target_f:
            x = F.pad(x, (0, 0, 0, target_f - x.shape[2]))
        elif x.shape[2] > target_f:
            x = x[:, :, :target_f, :]

        if x.shape[3] < target_t:
            x = F.pad(x, (0, target_t - x.shape[3]))
        elif x.shape[3] > target_t:
            x = x[:, :, :, :target_t]

        return self.model(x)

    def compute_loss(self, pred, target):
        """Compute spectrogram reconstruction loss.
        Pads/crops target to match pred shape (model may change dimensions due to patching).
        """
        if target.shape != pred.shape:
            _, _, pf, pt = pred.shape
            if target.shape[2] < pf:
                target = F.pad(target, (0, 0, 0, pf - target.shape[2]))
            elif target.shape[2] > pf:
                target = target[:, :, :pf, :]
            if target.shape[3] < pt:
                target = F.pad(target, (0, pt - target.shape[3]))
            elif target.shape[3] > pt:
                target = target[:, :, :, :pt]

        loss = self.loss_fn(pred, target)
        return loss

    def _prepare_batch(self, x, y):
        """
        OPTIMIZED: Simplified batch preparation.

        Returns:
            x: (B*Ctx, 1, F, T) - ready for model
            y: (B*Ctx, 1, F, T) - ready for loss
            B: original batch size
            Ctx: context length
        """
        if x.dim() == 3:
            return x.unsqueeze(1), y.unsqueeze(1), x.shape[0], 1
        elif x.dim() == 4:
            B, Ctx, F, T = x.shape
            x = x.contiguous().view(B * Ctx, 1, F, T)
            y = y.contiguous().view(B * Ctx, 1, F, T)
            return x, y, B, Ctx
        else:
            raise ValueError(f"Unexpected input dim: {x.dim()}")

    def training_step(self, batch, batch_idx):
        x, y = batch  # studio (input), live (target)
        x, y, B, Ctx = self._prepare_batch(x, y)

        y_pred = self(x)

        loss = self.compute_loss(y_pred, y)

        current_lr = self.optimizers().param_groups[0]["lr"]
        self.log("train/lr", current_lr, prog_bar=True)
        self.log("train/loss", loss, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x, y, B, Ctx = self._prepare_batch(x, y)

        y_pred = self(x)

        loss = self.compute_loss(y_pred, y)

        self.log("val/loss", loss, prog_bar=True)

        if batch_idx == 0:
            self.validation_outputs.append(
                {
                    "input": x.detach().cpu(),
                    "target": y.detach().cpu(),
                    "output": y_pred.detach().cpu(),
                }
            )

        return loss

    def on_before_optimizer_step(self, optimizer):
        if self.global_step % 10 == 0:
            total_norm = torch.nn.utils.clip_grad_norm_(
                self.parameters(),
                max_norm=float("inf"),
                norm_type=2,
            )
            self.log("train/grad_norm", total_norm, on_step=True, on_epoch=False)

    def on_validation_epoch_end(self):
        """Clean up validation outputs."""
        self.validation_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            betas=(0.9, 0.98),
            weight_decay=1e-6,
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=30,
            min_lr=1e-8,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "monitor": "val/loss",
            },
        }


def train(args=None):
    """Main training function."""

    if args is None:
        args = parse_args()

    learning_rate = args.learning_rate
    batch_size = args.batch_size
    max_epochs = args.max_epochs

    pl.seed_everything(42)

    datamodule = StudioLiveDataModule(
        studio_dir=args.studio_dir,
        live_dir=args.live_dir,
        batch_size=batch_size,
        sr=args.sample_rate,
        segment_duration=args.segment_duration,
        context_length=args.context_length,
        train_split=args.train_split,
        persistent_workers=True,
        num_workers=args.num_workers,
        development_mode=args.development_mode,
    )

    time_frames = int(
        (args.sample_rate * args.segment_duration) / 512
    )  # 22050 * 0.5 / 512 = ~22 frames for 0.5s segments with hop_length=512
    patch_size = args.patch_size
    input_tdim = (time_frames // patch_size) * patch_size
    if input_tdim < patch_size:
        input_tdim = patch_size

    model = LiveifyModel(
        input_fdim=128,  # n_mels from dataset
        input_tdim=input_tdim,
        patch_size=(patch_size, patch_size),
        embed_dim=args.embed_dim,
        num_transformer_layers=args.num_transformer_layers,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        gru_layers=args.gru_layers,
        dropout=args.dropout,
        attention_dropout=args.attention_dropout,
        in_channels=1,
        out_channels=1,
    )

    lightning_module = LiveifyLightningModule(
        model=model,
        learning_rate=learning_rate,
        sample_rate=args.sample_rate,
    )

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="best",
        monitor="val/loss",
        mode="min",
        save_top_k=1,
        save_last=True,
    )

    early_stop_callback = EarlyStopping(
        monitor="val/loss",
        patience=args.patience,
        mode="min",
        verbose=True,
    )

    logger = TensorBoardLogger(
        save_dir=args.log_dir,
        name="liveify",
    )

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        devices=1,
        precision=args.precision,
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=logger,
        log_every_n_steps=1,
        gradient_clip_val=1,
        gradient_clip_algorithm="norm",
        accumulate_grad_batches=args.accumulate_grad_batches,
    )

    print("\n" + "=" * 50)
    if args.resume_from:
        print(f"Resuming training from: {args.resume_from}")
    else:
        print("Starting training...")
    print("=" * 50 + "\n")

    trainer.fit(lightning_module, datamodule, ckpt_path=args.resume_from)

    print("\n" + "=" * 50)
    print("Training completed!")
    print(f"Best model checkpoint: {checkpoint_callback.best_model_path}")
    print("=" * 50 + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Train Liveify model")

    parser.add_argument(
        "--studio_dir",
        type=str,
        default="./dataset/studio",
        help="Path to studio recordings",
    )
    parser.add_argument(
        "--live_dir", type=str, default="./dataset/live", help="Path to live recordings"
    )
    parser.add_argument(
        "--sample_rate", type=int, default=22050, help="Audio sample rate"
    )
    parser.add_argument(
        "--segment_duration",
        type=float,
        default=0.5,
        help="Segment duration in seconds (aligned at 5s, chopped to this)",
    )
    parser.add_argument(
        "--context_length",
        type=int,
        default=64,
        help="Number of consecutive segments for LSTM temporal context",
    )
    parser.add_argument(
        "--train_split", type=float, default=0.8, help="Train/val split ratio"
    )

    parser.add_argument(
        "--batch_size", type=int, default=8, help="Batch size for training"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-6, help="Learning rate"
    )
    parser.add_argument(
        "--max_epochs", type=int, default=100, help="Maximum number of epochs"
    )
    parser.add_argument(
        "--patience", type=int, default=10, help="Early stopping patience"
    )
    parser.add_argument(
        "--num_workers", type=int, default=10, help="Number of dataloader workers"
    )
    parser.add_argument(
        "--accumulate_grad_batches",
        type=int,
        default=1,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="bf16-mixed",
        choices=["32", "16", "bf16", "16-mixed", "bf16-mixed"],
        help="Training precision",
    )

    parser.add_argument(
        "--embed_dim",
        type=int,
        default=512,
        help="Embedding dimension for transformer",
    )
    parser.add_argument(
        "--num_transformer_layers",
        type=int,
        default=8,
        help="Number of transformer encoder layers",
    )
    parser.add_argument(
        "--num_heads",
        type=int,
        default=8,
        help="Number of attention heads",
    )
    parser.add_argument(
        "--mlp_ratio",
        type=float,
        default=4.0,
        help="MLP hidden dimension ratio",
    )
    parser.add_argument(
        "--gru_layers",
        type=int,
        default=2,
        help="Number of GRU layers",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout rate",
    )
    parser.add_argument(
        "--attention_dropout",
        type=float,
        default=0.1,
        help="Attention dropout rate",
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        default=16,
        help="Patch size for Vision Transformer",
    )

    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="./checkpoints",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--log_dir", type=str, default="./logs", help="Directory to save logs"
    )

    parser.add_argument(
        "--development_mode",
        action="store_true",
        help="Use only first song pair for fast iteration",
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
