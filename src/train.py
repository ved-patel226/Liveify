import torch

torch.set_float32_matmul_precision("high")

from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger


import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for saving figures

from model import LiveifyModel
from dataset_utils.dataset import StudioLiveDataModule
from augmentation import SpectrogramAugmentation


class SpectrogramLoss(nn.Module):
    """Loss for spectrogram-based models combining spectral convergence, SI-SDR, and phase-sensitive spectral approximation."""

    def __init__(
        self,
        si_sdr_weight=0,
        psa_weight=1,
        l1_weight=0.1,
    ):
        super().__init__()
        self.si_sdr_weight = si_sdr_weight
        self.psa_weight = psa_weight
        self.l1_weight = l1_weight

    def _l2_norm(self, s1, s2):
        """L2 norm between two signals."""
        norm = torch.sum(s1 * s2, -1, keepdim=True)
        return norm

    def _si_snr(self, s1, s2, eps=1e-8):
        """Scale-Invariant Signal-to-Noise Ratio."""
        s1_s2_norm = self._l2_norm(s1, s2)
        s2_s2_norm = self._l2_norm(s2, s2)
        s_target = s1_s2_norm / (s2_s2_norm + eps) * s2
        e_noise = s1 - s_target
        target_norm = self._l2_norm(s_target, s_target)
        noise_norm = self._l2_norm(e_noise, e_noise)
        snr = 10 * torch.log10((target_norm) / (noise_norm + eps) + eps)
        return torch.mean(snr)

    def _loss_sisdr(self, inputs, targets):
        """SI-SDR loss (negative SI-SNR for minimization)."""
        return -self._si_snr(inputs, targets)

    def loss_phase_sensitive_spectral_approximation(self, enhance, target, mixture):
        """
        Phase-sensitive spectral approximation loss.
        Reference: Erdogan, Hakan, et al. "Phase-sensitive and recognition-boosted speech separation using deep recurrent neural networks." ICASSP 2015.
        """
        eps = nn.Parameter(
            data=torch.ones((1,), dtype=torch.float32) * 1e-9, requires_grad=False
        ).to(enhance.device)
        angle_mixture = torch.tanh(mixture[..., 1] / (mixture[..., 0] + eps))
        angle_target = torch.tanh(target[..., 1] / (target[..., 0] + eps))
        amplitude_enhance = torch.sqrt(enhance[..., 1] ** 2 + enhance[..., 0] ** 2)
        amplitude_target = torch.sqrt(target[..., 1] ** 2 + target[..., 0] ** 2)
        loss = amplitude_enhance - amplitude_target * torch.cos(
            angle_target - angle_mixture
        )
        loss = torch.mean(loss**2)  # mse
        return loss

    def forward(self, pred, target, mixture=None):
        """Compute combined loss on spectrograms.

        Args:
            pred: (batch, channels, freq, time)
            target: same shape as pred
            mixture: (batch, channels, freq, time), optional for PSA loss
        """
        loss = 0.0

        if self.si_sdr_weight > 0:
            pred_flat = pred.reshape(pred.shape[0], -1)
            target_flat = target.reshape(target.shape[0], -1)
            sisdr = self._loss_sisdr(pred_flat, target_flat)
            loss = loss + self.si_sdr_weight * sisdr

        if self.psa_weight > 0 and mixture is not None:
            psa = self.loss_phase_sensitive_spectral_approximation(
                pred, target, mixture
            )
            loss = loss + self.psa_weight * psa

        if self.l1_weight > 0:
            l1 = F.l1_loss(pred, target)
            loss = loss + self.l1_weight * l1

        return loss


class LiveifyLightningModule(pl.LightningModule):
    def __init__(
        self,
        model: LiveifyModel,
        learning_rate: float = 1e-4,
        sample_rate: int = 22050,
        use_augmentation: bool = True,
        aug_freq_mask: int = 20,
        aug_time_mask: int = 40,
        aug_noise_std: float = 0.01,
    ):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.sample_rate = sample_rate

        self.loss_fn = SpectrogramLoss(
            psa_weight=5.0,
            si_sdr_weight=0,
            l1_weight=1.0,
        )

        if use_augmentation:
            self.augmentation = SpectrogramAugmentation(
                freq_mask_param=aug_freq_mask,
                time_mask_param=aug_time_mask,
                num_freq_masks=2,
                num_time_masks=2,
                noise_std=aug_noise_std,
                p=0.5,
            )
        else:
            self.augmentation = None

        self.validation_outputs = []
        self.saved_input_target = False  # Track if we've saved input/target once
        self.sample_for_visualization = None  # Store a sample for visualization

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
        Prepare batch by concatenating context segments along the time axis.

        Returns:
            x: (B, 1, F, Ctx*T) - context segments concatenated in time
            y: (B, 1, F, Ctx*T) - context segments concatenated in time
            B: original batch size
            Ctx: context length
        """
        if x.dim() == 3:
            # (B, F, T) -> (B, 1, F, T)
            return x.unsqueeze(1), y.unsqueeze(1), x.shape[0], 1
        elif x.dim() == 4:
            # (B, Ctx, F, T) -> (B, 1, F, Ctx*T)
            B, Ctx, F, T = x.shape
            x = x.permute(0, 2, 1, 3).contiguous().view(B, 1, F, Ctx * T)
            y = y.permute(0, 2, 1, 3).contiguous().view(B, 1, F, Ctx * T)
            return x, y, B, Ctx
        else:
            raise ValueError(f"Unexpected input dim: {x.dim()}")

    def training_step(self, batch, batch_idx):
        x, y = batch  # studio (input), live (target)
        x, y, B, Ctx = self._prepare_batch(x, y)

        if self.augmentation is not None:
            x = self.augmentation(x)

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

        if batch_idx == 0 and self.sample_for_visualization is None:
            self.sample_for_visualization = {
                "input": x[0:1].detach().cpu(),
                "target": y[0:1].detach().cpu(),
            }

        if batch_idx == 0:
            if self.sample_for_visualization is not None:
                self.sample_for_visualization["output"] = y_pred[0:1].detach().cpu()

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
        """Save spectrograms every 10 epochs."""
        if self.sample_for_visualization is None:
            return

        current_epoch = self.current_epoch

        if not self.saved_input_target:
            self._save_spectrogram(
                self.sample_for_visualization["input"][0, 0].numpy(),
                "input_studio",
                "Studio Recording (Input)",
            )
            self._save_spectrogram(
                self.sample_for_visualization["target"][0, 0].numpy(),
                "target_live",
                "Live Recording (Target)",
            )
            self.saved_input_target = True

        if current_epoch % 10 == 0 and "output" in self.sample_for_visualization:
            self._save_spectrogram(
                self.sample_for_visualization["output"][0, 0].float().numpy(),
                f"output_epoch_{current_epoch:04d}",
                f"Model Prediction - Epoch {current_epoch}",
            )

    def _save_spectrogram(self, spec, filename, title):
        """Save a single spectrogram to file."""
        output_dir = Path("./spectrograms")
        output_dir.mkdir(exist_ok=True)

        fig, ax = plt.subplots(figsize=(12, 4))

        im = ax.imshow(
            spec, aspect="auto", origin="lower", cmap="viridis", vmin=-1, vmax=1
        )
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_ylabel("Mel Frequency Bin")
        ax.set_xlabel("Time Frame")
        plt.colorbar(im, ax=ax, label="Normalized Magnitude")

        plt.tight_layout()

        save_path = output_dir / f"{filename}.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        print(f"Saved spectrogram: {save_path}")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            betas=(0.9, 0.98),
            weight_decay=1e-3,
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=300,
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

    time_frames_per_segment = int(
        (args.sample_rate * args.segment_duration) / 512
    )  # 22050 * 0.5 / 512 = ~22 frames per segment with hop_length=512
    patch_size = args.patch_size

    total_time_frames = time_frames_per_segment * args.context_length
    input_tdim = (total_time_frames // patch_size) * patch_size
    if input_tdim < patch_size:
        input_tdim = patch_size

    print(f"Spectrogram config:")
    print(f"  Frames per segment: {time_frames_per_segment}")
    print(f"  Context length: {args.context_length}")
    print(f"  Total time frames: {total_time_frames} -> input_tdim: {input_tdim}")
    print(
        f"  Patches: {256 // patch_size} freq x {input_tdim // patch_size} time = {(256 // patch_size) * (input_tdim // patch_size)} total"
    )

    model = LiveifyModel(
        input_fdim=256,  # n_mels from dataset
        input_tdim=input_tdim,
        patch_size=(patch_size, patch_size),
        embed_dim=args.embed_dim,
        num_transformer_layers=args.num_transformer_layers,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
        attention_dropout=args.attention_dropout,
        in_channels=1,
        out_channels=1,
    )

    lightning_module = LiveifyLightningModule(
        model=model,
        learning_rate=learning_rate,
        sample_rate=args.sample_rate,
        use_augmentation=args.use_augmentation,
        aug_freq_mask=args.aug_freq_mask,
        aug_time_mask=args.aug_time_mask,
        aug_noise_std=args.aug_noise_std,
    )

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    monitor_metric = "train/loss" if args.train_split >= 1.0 else "val/loss"

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="best",
        monitor=monitor_metric,
        mode="min",
        save_top_k=1,
        save_last=True,
    )

    # early_stop_callback = EarlyStopping(
    #     monitor="val/loss",
    #     patience=args.patience,
    #     mode="min",
    #     verbose=True,
    # )

    if args.logger == "wandb":
        logger = WandbLogger(
            project="liveify",
            save_dir=args.log_dir,
            log_model=False,
        )
    else:
        logger = TensorBoardLogger(
            save_dir=args.log_dir,
            name="liveify",
        )

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        devices=1,
        precision=args.precision,
        callbacks=[checkpoint_callback],
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
        help="Number of consecutive segments concatenated along time axis for temporal context",
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
        "--patience", type=int, default=1000, help="Early stopping patience"
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
        default=4,
        help="Number of transformer encoder layers",
    )
    parser.add_argument(
        "--num_heads",
        type=int,
        default=4,
        help="Number of attention heads",
    )
    parser.add_argument(
        "--mlp_ratio",
        type=float,
        default=4.0,
        help="MLP hidden dimension ratio",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.3,
        help="Dropout rate",
    )
    parser.add_argument(
        "--attention_dropout",
        type=float,
        default=0.3,
        help="Attention dropout rate",
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        default=16,
        help="Patch size for Vision Transformer",
    )

    parser.add_argument(
        "--use_augmentation",
        action="store_true",
        default=True,
        help="Use data augmentation during training",
    )
    parser.add_argument(
        "--no_augmentation",
        dest="use_augmentation",
        action="store_false",
        help="Disable data augmentation",
    )
    parser.add_argument(
        "--aug_freq_mask",
        type=int,
        default=20,
        help="Maximum frequency mask width for SpecAugment",
    )
    parser.add_argument(
        "--aug_time_mask",
        type=int,
        default=40,
        help="Maximum time mask width for SpecAugment",
    )
    parser.add_argument(
        "--aug_noise_std",
        type=float,
        default=0.01,
        help="Standard deviation of Gaussian noise to add",
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
        "--logger",
        type=str,
        default="tensorboard",
        choices=["tensorboard", "wandb"],
        help="Logger to use for experiment tracking",
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
