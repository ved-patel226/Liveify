import torch

torch.set_float32_matmul_precision("high")

from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger


import torch.nn as nn
import pytorch_lightning as pl
from pathlib import Path
import argparse

from model import LiveifyModel
from dataset.dataset import StudioLiveDataModule


def normalize_audio_for_logging(audio: torch.Tensor) -> torch.Tensor:
    """
    Normalize audio to [-1, 1] range for safe TensorBoard logging.
    Clamps values to avoid warnings.
    """
    # Ensure we're working with float tensors
    audio = audio.float()

    # Get max absolute value per batch element
    batch_size = audio.shape[0]
    for i in range(batch_size):
        max_val = torch.max(torch.abs(audio[i]))
        if max_val > 0:
            audio[i] = audio[i] / (max_val + 1e-7)

    # Hard clamp to [-1, 1] to avoid floating point artifacts
    return torch.clamp(audio, -1.0, 1.0)


class SpectralLoss(nn.Module):
    """Multi-scale spectral loss for audio quality."""

    def __init__(self, n_ffts=[512, 1024, 2048], hop_lengths=None):
        super().__init__()
        self.n_ffts = n_ffts
        self.hop_lengths = hop_lengths or [n // 4 for n in n_ffts]

    def forward(self, pred, target):
        loss = 0.0
        for n_fft, hop_length in zip(self.n_ffts, self.hop_lengths):
            pred_stft = torch.stft(
                pred,
                n_fft=n_fft,
                hop_length=hop_length,
                return_complex=True,
                normalized=True,
            )
            target_stft = torch.stft(
                target,
                n_fft=n_fft,
                hop_length=hop_length,
                return_complex=True,
                normalized=True,
            )

            pred_mag = torch.abs(pred_stft)
            target_mag = torch.abs(target_stft)
            loss += nn.functional.l1_loss(pred_mag, target_mag)

            loss += nn.functional.l1_loss(
                torch.log(pred_mag + 1e-5), torch.log(target_mag + 1e-5)
            )

        return loss / len(self.n_ffts)


class LiveifyLightningModule(pl.LightningModule):
    def __init__(
        self,
        model: LiveifyModel,
        learning_rate: float = 1e-4,
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

        self.input_audio_logged = False
        self.validation_outputs = []

        self.save_hyperparameters(ignore=["model"])

    def forward(self, x):
        return self.model(x)

    def compute_loss(self, pred, target):
        """Compute combined time-domain and spectral loss."""
        time_loss = self.time_loss(pred, target)
        spectral_loss = self.spectral_loss(pred, target)

        total_loss = (
            self.time_loss_weight * time_loss
            + self.spectral_loss_weight * spectral_loss
        )

        return total_loss, time_loss, spectral_loss

    def training_step(self, batch, batch_idx):
        x, y = batch  # studio, live
        y_pred = self(x)

        total_loss, time_loss, spectral_loss = self.compute_loss(y_pred, y)

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

        # Store outputs for audio logging
        if batch_idx == 0:  # Only log from first batch to avoid too many samples
            self.validation_outputs.append(
                {
                    "input": x.detach().cpu(),
                    "target": y.detach().cpu(),
                    "output": y_pred.detach().cpu(),
                }
            )

        return total_loss

    def on_validation_epoch_end(self):
        """Log audio samples at the end of each validation epoch."""
        if not self.validation_outputs:
            return

        outputs = self.validation_outputs[0]
        input_audio = outputs["input"]
        target_audio = outputs["target"]
        output_audio = outputs["output"]

        # Normalize all audio for safe logging
        input_audio = normalize_audio_for_logging(input_audio)
        target_audio = normalize_audio_for_logging(target_audio)
        output_audio = normalize_audio_for_logging(output_audio)

        if not self.input_audio_logged and self.logger is not None:
            for i in range(min(3, input_audio.shape[0])):
                self.logger.experiment.add_audio(
                    f"audio/input_sample_{i}",
                    input_audio[i].unsqueeze(0),
                    global_step=self.global_step,
                    sample_rate=self.sample_rate,
                )
                self.logger.experiment.add_audio(
                    f"audio/target_sample_{i}",
                    target_audio[i].unsqueeze(0),
                    global_step=self.global_step,
                    sample_rate=self.sample_rate,
                )
            self.input_audio_logged = True

        if self.logger is not None:
            for i in range(min(3, output_audio.shape[0])):
                self.logger.experiment.add_audio(
                    f"audio/output_sample_{i}",
                    output_audio[i].unsqueeze(0),
                    global_step=self.global_step,
                    sample_rate=self.sample_rate,
                )

        self.validation_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=1e-4,
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=5,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss",
            },
        }


def train(args):
    """Main training function."""

    pl.seed_everything(42)

    datamodule = StudioLiveDataModule(
        studio_dir=args.studio_dir,
        live_dir=args.live_dir,
        batch_size=args.batch_size,
        sr=args.sample_rate,
        segment_duration=args.segment_duration,
        train_split=args.train_split,
        num_workers=args.num_workers,
        development_mode=args.development_mode,
    )

    model = LiveifyModel(
        input_sr=int(args.sample_rate * args.segment_duration),
        output_sr=int(args.sample_rate * args.segment_duration),
        hidden_channels=128,
        encoder_strides=[8, 4, 4],
        transformer_dim=(512 + 256) // 2,
        num_heads=8,
        num_layers=4,
        lr=args.learning_rate,
    )

    lightning_module = LiveifyLightningModule(
        model=model,
        learning_rate=args.learning_rate,
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

    logger = TensorBoardLogger(
        save_dir=args.log_dir,
        name="liveify",
    )

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
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
        default=4.0,
        help="Segment duration in seconds",
    )
    parser.add_argument(
        "--train_split", type=float, default=0.8, help="Train/val split ratio"
    )

    parser.add_argument(
        "--batch_size", type=int, default=8, help="Batch size for training"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-4, help="Learning rate"
    )
    parser.add_argument(
        "--max_epochs", type=int, default=100, help="Maximum number of epochs"
    )
    parser.add_argument(
        "--patience", type=int, default=100, help="Early stopping patience"
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of dataloader workers"
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
        default="32",
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
        default=1.0,
        help="Weight for spectral loss",
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
