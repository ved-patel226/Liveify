import torch

torch.set_float32_matmul_precision("high")

from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger


import torch.nn as nn
import pytorch_lightning as pl
from pathlib import Path
import argparse
import wandb

from model import LiveifyModel
from dataset.dataset import StudioLiveDataModule


class SpectralLoss(nn.Module):
    """Multi-scale spectral loss for audio quality."""

    def __init__(self, n_ffts=[512, 1024, 2048], hop_lengths=None):
        super().__init__()
        self.n_ffts_default = n_ffts
        self.hop_lengths_default = hop_lengths or [n // 4 for n in n_ffts]

    def _get_adaptive_params(self, seq_len):
        """Adapt FFT sizes to sequence length. For short audio, use smaller FFTs."""
        # for very short sequences (< 512 samples), scale down the FFT sizes
        if seq_len < 512:
            # use FFT sizes that fit within sequence length
            scale = max(1, seq_len // 128)
            n_ffts = [64 * scale, 128 * scale]
            hop_lengths = [n // 4 for n in n_ffts]
        else:
            n_ffts = self.n_ffts_default
            hop_lengths = self.hop_lengths_default

        n_ffts = [min(n_fft, seq_len // 2) for n_fft in n_ffts]
        n_ffts = [max(16, n_fft) for n_fft in n_ffts]  # minimum 16
        hop_lengths = [min(n // 4, seq_len // 8) for n in n_ffts]
        hop_lengths = [max(1, h) for h in hop_lengths]

        return n_ffts, hop_lengths

    def forward(self, pred, target):
        seq_len = pred.shape[-1]
        n_ffts, hop_lengths = self._get_adaptive_params(seq_len)

        pred_float = (
            pred.float() if pred.dtype in [torch.bfloat16, torch.float16] else pred
        )
        target_float = (
            target.float()
            if target.dtype in [torch.bfloat16, torch.float16]
            else target
        )

        loss = 0.0
        for n_fft, hop_length in zip(n_ffts, hop_lengths):
            if n_fft < 2:  # skip if FFT size is too small
                continue

            window = torch.hann_window(n_fft, device=pred.device, dtype=torch.float32)

            pred_stft = torch.stft(
                pred_float,
                n_fft=n_fft,
                hop_length=hop_length,
                window=window,
                return_complex=True,
                normalized=True,
            )
            target_stft = torch.stft(
                target_float,
                n_fft=n_fft,
                hop_length=hop_length,
                window=window,
                return_complex=True,
                normalized=True,
            )

            pred_mag = torch.abs(pred_stft)
            target_mag = torch.abs(target_stft)
            loss += nn.functional.l1_loss(pred_mag, target_mag)

            loss += nn.functional.l1_loss(
                torch.log(pred_mag + 1e-5), torch.log(target_mag + 1e-5)
            )

        num_scales = len([n for n in n_ffts if n >= 2])

        final_loss = loss / max(1, num_scales)
        if pred.dtype in [torch.bfloat16, torch.float16]:
            final_loss = final_loss.to(pred.dtype)

        return final_loss


class LiveifyLightningModule(pl.LightningModule):
    def __init__(
        self,
        model: LiveifyModel,
        learning_rate: float = 3.5e-6,
        time_loss_weight: float = 1.0,
        spectral_loss_weight: float = 1.0,
        sample_rate: int = 22050,
    ):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.time_loss_weight = time_loss_weight
        self.spectral_loss_weight = spectral_loss_weight
        self.sample_rate = sample_rate

        self.time_loss = nn.L1Loss()
        self.spectral_loss = SpectralLoss()

        self.validation_outputs = []

        self.save_hyperparameters(ignore=["model"])

    def forward(self, x):
        return self.model(x)

    def compute_loss(self, pred, target):
        """Compute combined time-domain and spectral loss.
        Handles both 2D (batch, samples) and 3D (batch, context, samples) tensors.
        """
        if pred.dim() == 3:
            B, C, S = pred.shape
            pred_flat = pred.reshape(B * C, S)
            target_flat = target.reshape(B * C, S)
        else:
            pred_flat = pred
            target_flat = target

        time_loss = self.time_loss(pred_flat, target_flat)
        spectral_loss = self.spectral_loss(pred_flat, target_flat)

        self.log("debug/time_loss_raw", time_loss, prog_bar=False)
        self.log("debug/spectral_loss_raw", spectral_loss, prog_bar=False)

        total_loss = (
            self.time_loss_weight * time_loss
            + self.spectral_loss_weight * spectral_loss
        )

        return total_loss, time_loss, spectral_loss

    def training_step(self, batch, batch_idx):
        x, y = batch  # studio, live
        y_pred = self(x)

        total_loss, time_loss, spectral_loss = self.compute_loss(y_pred, y)

        current_lr = self.optimizers().param_groups[0]["lr"]
        self.log("train/lr", current_lr, prog_bar=True)

        self.log("train/loss", total_loss, prog_bar=True)
        self.log("train/time_loss", time_loss)
        self.log("train/spectral_loss", spectral_loss)

        return total_loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)

        total_loss, time_loss, spectral_loss = self.compute_loss(y_pred, y)

        self.log("val/loss", total_loss, prog_bar=True)
        self.log("val/time_loss", time_loss)
        self.log("val/spectral_loss", spectral_loss)

        if batch_idx == 0:
            self.validation_outputs.append(
                {
                    "input": x.detach().cpu(),
                    "target": y.detach().cpu(),
                    "output": y_pred.detach().cpu(),
                }
            )

        return total_loss

    def on_before_optimizer_step(self, optimizer):
        """Monitor gradient norms."""
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm**0.5

        self.log("grad_norm", total_norm, prog_bar=True)

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
            patience=5,
            min_lr=1e-8,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "monitor": "train/loss",
            },
        }


def train(args=None):
    """Main training function."""

    if not wandb.run:
        wandb.init(project="liveify")

    if wandb.run and hasattr(wandb.config, "learning_rate"):
        learning_rate = wandb.config.learning_rate
        batch_size = getattr(wandb.config, "batch_size", 40)
        max_epochs = getattr(wandb.config, "max_epochs", 10)
        if args is None:
            args = parse_args()
    else:
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

    model = LiveifyModel(
        input_sr=int(args.sample_rate * args.segment_duration),
        output_sr=int(args.sample_rate * args.segment_duration),
        hidden_channels=256,
        encoder_strides=[8, 4, 4],
        transformer_dim=256,
        num_heads=8,
        num_layers=6,
        context_length=args.context_length,
        lr=1e-5,
    )

    lightning_module = LiveifyLightningModule(
        model=model,
        learning_rate=learning_rate,
        time_loss_weight=args.time_loss_weight,
        spectral_loss_weight=args.spectral_loss_weight,
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

    logger = WandbLogger(
        project="liveify",
        save_dir=args.log_dir,
        log_model="all",
    )

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        devices=1,
        precision=args.precision,
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=logger,
        log_every_n_steps=10,
        gradient_clip_val=1.0,
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
        "--learning_rate", type=float, default=1e-5, help="Learning rate"
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
        default=16,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="bf16",
        choices=["32", "16", "bf16"],
        help="Training precision",
    )

    parser.add_argument(
        "--time_loss_weight",
        type=float,
        default=1.0,
        help="Weight for time-domain loss",
    )
    parser.add_argument(
        "--spectral_loss_weight",
        type=float,
        default=0.25,
        help="Weight for spectral loss (default 0.25 since spectral loss is ~10x time loss)",
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
    if wandb.run:
        train()
    else:
        args = parse_args()
        train(args)
