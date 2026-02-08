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
    audio = audio.float()

    batch_size = audio.shape[0]
    for i in range(batch_size):
        max_val = torch.max(torch.abs(audio[i]))
        if max_val > 0:
            audio[i] = audio[i] / (max_val + 1e-7)

    return torch.clamp(audio, -1.0, 1.0)


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

        loss = 0.0
        for n_fft, hop_length in zip(n_ffts, hop_lengths):
            if n_fft < 2:  # skip if FFT size is too small
                continue

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

        num_scales = len([n for n in n_ffts if n >= 2])
        return loss / max(1, num_scales)


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

        self.input_audio_logged = False
        self.validation_outputs = []

        self.save_hyperparameters(ignore=["model"])

    def forward(self, x):
        return self.model(x)

    def compute_loss(self, pred, target):
        """Compute combined time-domain and spectral loss."""
        time_loss = self.time_loss(pred, target)
        spectral_loss = self.spectral_loss(pred, target)

        self.log("debug/time_loss_raw", time_loss, prog_bar=False)
        self.log("debug/spectral_loss_raw", spectral_loss, prog_bar=False)

        total_loss = (
            self.time_loss_weight * time_loss
            + self.spectral_loss_weight * 0.1 * spectral_loss
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
        """Log audio samples at the end of each validation epoch."""
        if not self.validation_outputs:
            return

        outputs = self.validation_outputs[0]
        input_audio = outputs["input"]
        target_audio = outputs["target"]
        output_audio = outputs["output"]

        input_audio = normalize_audio_for_logging(input_audio)
        target_audio = normalize_audio_for_logging(target_audio)
        output_audio = normalize_audio_for_logging(output_audio)

        segment_duration = input_audio.shape[1] / self.sample_rate

        if self.logger is not None:
            if segment_duration < 5.0:
                target_samples = int(
                    5.0 * self.sample_rate
                )  # 5 seconds worth of samples

                batch_indices = list(range(min(input_audio.shape[0], 50)))

                segment_data = []
                for i in batch_indices:
                    try:
                        dataset = self.trainer.datamodule.val_dataset.dataset
                        actual_idx = (
                            self.trainer.datamodule.val_dataset.indices[i]
                            if hasattr(self.trainer.datamodule.val_dataset, "indices")
                            else i
                        )

                        if actual_idx < len(dataset.pairs):
                            pair_info = dataset.pairs[actual_idx]
                            song_key = (
                                f"{pair_info['studio_name']}_{pair_info['live_name']}"
                            )
                            segment_idx = pair_info.get("segment_idx", 0)
                            sub_segment_idx = pair_info.get("sub_segment_idx", 0)
                            sort_key = (song_key, segment_idx, sub_segment_idx)
                        else:
                            sort_key = (f"unknown_{i}", 0, 0)
                    except:
                        sort_key = (f"fallback_{i}", 0, 0)

                    segment_data.append(
                        (input_audio[i], target_audio[i], output_audio[i], sort_key)
                    )

                segment_data.sort(key=lambda x: x[3])

                sorted_input = [item[0] for item in segment_data]
                sorted_target = [item[1] for item in segment_data]
                sorted_output = [item[2] for item in segment_data]

                segments_needed = min(
                    target_samples // input_audio.shape[1], len(sorted_input)
                )

                stitched_input = torch.cat(sorted_input[:segments_needed]).flatten()[
                    :target_samples
                ]
                stitched_target = torch.cat(sorted_target[:segments_needed]).flatten()[
                    :target_samples
                ]
                stitched_output = torch.cat(sorted_output[:segments_needed]).flatten()[
                    :target_samples
                ]

                #  stitched audio (single 5-second sample)
                if not self.input_audio_logged:
                    self.logger.experiment.add_audio(
                        "audio/input_stitched_5s",
                        stitched_input.unsqueeze(0),
                        global_step=self.global_step,
                        sample_rate=self.sample_rate,
                    )
                    self.logger.experiment.add_audio(
                        "audio/target_stitched_5s",
                        stitched_target.unsqueeze(0),
                        global_step=self.global_step,
                        sample_rate=self.sample_rate,
                    )
                    self.input_audio_logged = True

                self.logger.experiment.add_audio(
                    "audio/output_stitched_5s",
                    stitched_output.unsqueeze(0),
                    global_step=self.global_step,
                    sample_rate=self.sample_rate,
                )

                for i in range(min(3, input_audio.shape[0])):
                    if not self.input_audio_logged:
                        self.logger.experiment.add_audio(
                            f"audio/input_segment_{i}",
                            input_audio[i].unsqueeze(0),
                            global_step=self.global_step,
                            sample_rate=self.sample_rate,
                        )
                        self.logger.experiment.add_audio(
                            f"audio/target_segment_{i}",
                            target_audio[i].unsqueeze(0),
                            global_step=self.global_step,
                            sample_rate=self.sample_rate,
                        )

                    self.logger.experiment.add_audio(
                        f"audio/output_segment_{i}",
                        output_audio[i].unsqueeze(0),
                        global_step=self.global_step,
                        sample_rate=self.sample_rate,
                    )

                if not self.input_audio_logged:
                    self.input_audio_logged = True

            else:
                if not self.input_audio_logged:
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
        encoder_strides=[2, 2],
        transformer_dim=512,
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
        default=0.02,
        help="Segment duration in seconds (aligned at 5s, chopped to this)",
    )
    parser.add_argument(
        "--train_split", type=float, default=0.8, help="Train/val split ratio"
    )

    parser.add_argument(
        "--batch_size", type=int, default=8, help="Batch size for training"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=3.5e-6, help="Learning rate"
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
