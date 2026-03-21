import torch

torch.set_float32_matmul_precision("high")

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pathlib import Path
import argparse
import io

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")

import librosa
import numpy as np

from model import LiveifyModel
from dataset_utils.dataset import StudioLiveDataModule
from augmentation import SpectrogramAugmentation


class SpectrogramLoss(nn.Module):
    """
    SIMPLIFIED Loss for mel spectrograms.
    Uses L1 (sparse differences) + L2 (overall structure).
    No PSA or SI-SNR since we're working with mel spectrograms, not complex spectrograms.
    """

    def __init__(self, l1_weight=1.0, l2_weight=1.0):
        super().__init__()
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight

    def forward(self, pred, target, mixture=None):
        """
        Args:
            pred: model output (batch, channels, freq, time)
            target: target spectrogram (batch, channels, freq, time)
            mixture: unused (for compatibility)
        Returns:
            scalar loss
        """
        loss = 0.0

        # L1 loss for sparse differences
        if self.l1_weight > 0:
            loss += self.l1_weight * F.l1_loss(pred, target)

        # L2 loss for overall structure
        if self.l2_weight > 0:
            loss += self.l2_weight * F.mse_loss(pred, target)

        return loss


def audio_to_mel_tensor(
    audio: torch.Tensor,
    sr: int,
    n_mels: int = 256,
    n_fft: int = 1024,
    hop_length: int = 512,
) -> torch.Tensor:
    """(B, slots, samples) -> (B, slots, 1, n_mels, T)"""
    B, slots, samples = audio.shape
    device = audio.device

    mel_fb = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels, fmax=sr // 2)
    mel_fb = torch.tensor(mel_fb, dtype=torch.float32, device=device)
    window = torch.hann_window(n_fft, device=device)

    specs = []
    for slot_idx in range(slots):
        slot_audio = audio[:, slot_idx, :]
        stft = torch.stft(
            slot_audio,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=n_fft,
            window=window,
            return_complex=True,
        )
        power = stft.abs() ** 2
        mel = torch.matmul(mel_fb, power)
        mel_db = 10.0 * torch.log10(mel.clamp(min=1e-9))
        mel_min = mel_db.amin(dim=(-2, -1), keepdim=True)
        mel_max = mel_db.amax(dim=(-2, -1), keepdim=True)
        mel_db = 2.0 * (mel_db - mel_min) / (mel_max - mel_min + 1e-8) - 1.0
        specs.append(mel_db.unsqueeze(1))

    return torch.stack(specs, dim=1)


def mel_to_audio_griffin_lim(
    mel_spec: np.ndarray,  # (F, T) normalised [-1, 1]
    sr: int,
    n_fft: int = 1024,
    hop_length: int = 512,
    n_iter: int = 64,
) -> np.ndarray:
    """
    Rough Griffin-Lim inversion of a normalised mel spectrogram.
    Good enough to hear whether the model is producing reasonable structure.
    """
    mel_db = (mel_spec + 1.0) * 40.0 - 80.0
    mel_power = np.power(10.0, mel_db / 10.0)

    n_mels = mel_spec.shape[0]
    mel_fb = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)
    mel_fb_inv = np.linalg.pinv(mel_fb)  # (n_fft//2+1, n_mels)
    linear_power = np.maximum(mel_fb_inv @ mel_power, 0.0)
    linear_mag = np.sqrt(linear_power)

    audio = librosa.griffinlim(
        linear_mag, n_iter=n_iter, hop_length=hop_length, win_length=n_fft
    )
    peak = np.abs(audio).max()
    if peak > 0:
        audio = audio / peak * 0.9
    return audio.astype(np.float32)


_SPEC_KW = dict(aspect="auto", origin="lower", cmap="magma", vmin=-1, vmax=1)
_DIFF_KW = dict(aspect="auto", origin="lower", cmap="RdBu_r", vmin=-1, vmax=1)


def _fig_to_numpy(fig) -> np.ndarray:
    """Render a matplotlib figure to an (H, W, 3) uint8 array."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    buf.seek(0)
    from PIL import Image

    img = np.array(Image.open(buf).convert("RGB"))
    buf.close()
    return img


def make_comparison_figure(
    studio: np.ndarray,
    output: np.ndarray,
    target: np.ndarray,
    epoch: int,
    sample_idx: int,
) -> plt.Figure:
    diff = output - target
    mae = np.abs(diff).mean()

    fig, axes = plt.subplots(1, 4, figsize=(22, 4))
    fig.suptitle(
        f"Epoch {epoch}  |  Sample {sample_idx}  |  MAE = {mae:.4f}",
        fontsize=11,
        fontweight="bold",
    )

    panels = [
        (studio, "Studio (input)", _SPEC_KW),
        (output, "Model output", _SPEC_KW),
        (target, "Live target", _SPEC_KW),
        (diff, "Signed diff (out-tgt)", _DIFF_KW),
    ]
    for ax, (data, title, kw) in zip(axes, panels):
        im = ax.imshow(data, **kw)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Time frame")
        ax.set_ylabel("Mel bin")
        plt.colorbar(im, ax=ax, shrink=0.8)

    fig.tight_layout()
    return fig


def make_context_strip_figure(
    context_slots: np.ndarray,  # (S, F, T)
    num_context: int,
    epoch: int,
    sample_idx: int,
) -> plt.Figure:
    S, F, T = context_slots.shape
    fig, axes = plt.subplots(1, S, figsize=(max(S * 2, 6), 3))
    if S == 1:
        axes = [axes]

    fig.suptitle(
        f"Context strip  |  Epoch {epoch}  |  Sample {sample_idx}  "
        f"({num_context} valid context + 1 target)",
        fontsize=9,
        fontweight="bold",
    )

    first_valid = S - 1 - num_context

    for slot_i, ax in enumerate(axes):
        ax.imshow(context_slots[slot_i], **_SPEC_KW)
        ax.set_xticks([])
        ax.set_yticks([])

        if slot_i == S - 1:
            label, color = "TARGET", "gold"
        elif slot_i >= first_valid:
            label, color = f"ctx {slot_i - first_valid}", "limegreen"
        else:
            label, color = "PAD", "tomato"

        ax.set_title(label, fontsize=7, color=color, fontweight="bold")
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(2)

    fig.tight_layout()
    return fig


def make_error_histogram_figure(
    outputs: list,
    targets: list,
    epoch: int,
) -> plt.Figure:
    """
    Distribution of per-pixel signed errors across all visualisation samples.

    What to look for:
      - Centred on 0 = unbiased predictions (good)
      - Shrinking spread over epochs = the model is learning
      - Peak at a non-zero value = systematic bias (bad)
      - All mass at 0 = mode collapse / model always predicts mean
    """
    errors = np.concatenate([(o - t).ravel() for o, t in zip(outputs, targets)])
    mean_e = errors.mean()
    std_e = errors.std()
    mae = np.abs(errors).mean()

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(
        errors, bins=100, range=(-2, 2), color="steelblue", alpha=0.75, density=True
    )
    ax.axvline(0, color="black", lw=1.5, ls="--", label="zero")
    ax.axvline(mean_e, color="tomato", lw=1.5, ls="-", label=f"mean = {mean_e:.4f}")
    ax.set_title(
        f"Error distribution  |  Epoch {epoch}  |  "
        f"μ={mean_e:.4f}  σ={std_e:.4f}  MAE={mae:.4f}",
        fontsize=10,
    )
    ax.set_xlabel("output − target (pixel value)")
    ax.set_ylabel("density")
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig


def make_output_stats_figure(
    outputs: list,
    targets: list,
    epoch: int,
) -> plt.Figure:
    """
    Bar chart: mean / std / min / max of model outputs vs live targets.

    Immediate mode-collapse detector: if output_std << target_std the model
    is predicting a near-constant spectrogram.
    """

    def stats(arrs):
        flat = np.concatenate([a.ravel() for a in arrs])
        return dict(
            mean=float(flat.mean()),
            std=float(flat.std()),
            min=float(flat.min()),
            max=float(flat.max()),
        )

    out_s = stats(outputs)
    tgt_s = stats(targets)
    keys = ["mean", "std", "min", "max"]
    x = np.arange(len(keys))
    w = 0.35

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(
        x - w / 2,
        [out_s[k] for k in keys],
        w,
        label="model output",
        color="steelblue",
        alpha=0.8,
    )
    ax.bar(
        x + w / 2,
        [tgt_s[k] for k in keys],
        w,
        label="live target",
        color="darkorange",
        alpha=0.8,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(keys)
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_title(f"Output vs target pixel statistics  |  Epoch {epoch}", fontsize=10)
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig


class LiveifyLightningModule(pl.LightningModule):
    def __init__(
        self,
        model: LiveifyModel,
        learning_rate: float = 1e-4,
        sample_rate: int = 22050,
        n_mels: int = 256,
        n_fft: int = 1024,
        hop_length: int = 512,
        use_augmentation: bool = True,
        aug_freq_mask: int = 20,
        aug_time_mask: int = 40,
        aug_noise_std: float = 0.02,
        # ----- vis -----
        viz_every_n_epochs: int = 5,
        viz_num_samples: int = 4,
        viz_log_audio: bool = True,
        viz_save_local: bool = True,
    ):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.viz_every_n_epochs = viz_every_n_epochs
        self.viz_num_samples = viz_num_samples
        self.viz_log_audio = viz_log_audio
        self.viz_save_local = viz_save_local

        # SIMPLIFIED LOSS: L1 + L2 only
        self.loss_fn = SpectrogramLoss(l1_weight=1.0, l2_weight=1.0)

        if use_augmentation:
            # LESS AGGRESSIVE AUGMENTATION
            # Reduce masking parameters and apply only 50% of time
            self.augmentation = SpectrogramAugmentation(
                freq_mask_param=aug_freq_mask,
                time_mask_param=aug_time_mask,
                num_freq_masks=2,
                num_time_masks=2,
                noise_std=aug_noise_std,
                p=0.5,  # apply augmentation 50% of the time
            )
        else:
            self.augmentation = None

        self._viz_samples: list = []
        self._viz_inputs_frozen: bool = False

        self.save_hyperparameters(ignore=["model"])

    def _batch_to_mel(self, audio: torch.Tensor) -> torch.Tensor:
        mel = audio_to_mel_tensor(
            audio,
            sr=self.sample_rate,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
        )
        target_f, target_t = self.model.input_fdim, self.model.input_tdim

        if mel.shape[3] < target_f:
            mel = F.pad(mel, (0, 0, 0, target_f - mel.shape[3]))
        elif mel.shape[3] > target_f:
            mel = mel[:, :, :, :target_f, :]
        if mel.shape[4] < target_t:
            mel = F.pad(mel, (0, target_t - mel.shape[4]))
        elif mel.shape[4] > target_t:
            mel = mel[:, :, :, :, :target_t]

        return mel

    def forward(self, x):
        return self.model(x)

    def compute_loss(self, pred, target_slots):
        target = target_slots[:, -1]
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
        return self.loss_fn(pred, target)

    def training_step(self, batch, batch_idx):
        studio_audio = batch["studio_audio"]
        live_audio = batch["live_audio"]

        x = self._batch_to_mel(studio_audio)
        y = self._batch_to_mel(live_audio)

        if self.augmentation is not None:
            B, S, C, F, T = x.shape
            x_flat = x.view(B * S, C, F, T)
            x_flat = self.augmentation(x_flat)
            x = x_flat.view(B, S, C, F, T)

        y_pred = self(x)
        loss = self.compute_loss(y_pred, y)

        lr = self.optimizers().param_groups[0]["lr"]
        self.log("train/lr", lr, prog_bar=True)
        self.log("train/loss", loss, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        studio_audio = batch["studio_audio"]
        live_audio = batch["live_audio"]

        x = self._batch_to_mel(studio_audio)
        y = self._batch_to_mel(live_audio)
        y_pred = self(x)
        loss = self.compute_loss(y_pred, y)

        self.log("val/loss", loss, prog_bar=True)

        B = x.shape[0]
        for b in range(B):
            global_idx = batch_idx * B + b
            if global_idx >= self.viz_num_samples:
                break

            out_np = y_pred[b, 0].detach().cpu().float().numpy()  # (F, T)

            if not self._viz_inputs_frozen:
                self._viz_samples.append(
                    {
                        "studio": x[b, -1, 0].detach().cpu().float().numpy(),  # (F, T)
                        "target": y[b, -1, 0].detach().cpu().float().numpy(),  # (F, T)
                        "context_all": x[b, :, 0]
                        .detach()
                        .cpu()
                        .float()
                        .numpy(),  # (S, F, T)
                        "num_context": int(batch["num_context"][b].item()),
                        "output": out_np,
                    }
                )
            elif global_idx < len(self._viz_samples):
                self._viz_samples[global_idx]["output"] = out_np

        return loss

    def _emit_figure(
        self, fig: plt.Figure, tag: str, epoch: int, is_wandb: bool
    ) -> None:
        if is_wandb:
            import wandb

            self.logger.experiment.log({tag: wandb.Image(fig)}, step=self.global_step)
        else:
            img = _fig_to_numpy(fig)  # (H, W, 3) uint8
            img_t = torch.from_numpy(img).permute(2, 0, 1)  # (3, H, W)
            self.logger.experiment.add_image(tag, img_t, global_step=epoch)

        if self.viz_save_local:
            out_dir = Path("./spectrograms") / f"epoch_{epoch:04d}"
            out_dir.mkdir(parents=True, exist_ok=True)
            safe = tag.replace("/", "_")
            fig.savefig(out_dir / f"{safe}.png", dpi=120, bbox_inches="tight")

        plt.close(fig)

    def _log_audio_wandb(self, epoch: int) -> None:
        """
        Invert mel spectrograms with Griffin-Lim and log as playable W&B audio.
        Lets you hear: does the output sound like the live recording or the studio?
        """
        import wandb

        audio_logs = {}
        for i, s in enumerate(self._viz_samples):
            for role in ("studio", "output", "target"):
                try:
                    wav = mel_to_audio_griffin_lim(
                        s[role],
                        sr=self.sample_rate,
                        n_fft=self.n_fft,
                        hop_length=self.hop_length,
                    )
                    audio_logs[f"viz/audio/sample_{i}/{role}"] = wandb.Audio(
                        wav,
                        sample_rate=self.sample_rate,
                        caption=f"Epoch {epoch} | sample {i} | {role}",
                    )
                except Exception as e:
                    print(f"[audio] Griffin-Lim failed: sample {i} / {role}: {e}")

        if audio_logs:
            self.logger.experiment.log(audio_logs, step=self.global_step)

    def on_before_optimizer_step(self, optimizer):
        if self.global_step % 10 == 0:
            total_norm = torch.nn.utils.clip_grad_norm_(
                self.parameters(),
                max_norm=float("inf"),
                norm_type=2,
            )
            self.log("train/grad_norm", total_norm, on_step=True, on_epoch=False)

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
            patience=120,
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
    if args is None:
        args = parse_args()

    pl.seed_everything(42)

    datamodule = StudioLiveDataModule(
        studio_dir=args.studio_dir,
        live_dir=args.live_dir,
        batch_size=args.batch_size,
        sr=args.sample_rate,
        segment_duration=args.segment_duration,
        context_length=args.context_length,
        train_split=args.train_split,
        num_workers=args.num_workers,
    )

    hop_length = 512
    time_frames_per_segment = int(
        (args.sample_rate * args.segment_duration) / hop_length
    )
    patch_size = args.patch_size

    input_tdim = (time_frames_per_segment // patch_size) * patch_size
    if input_tdim < patch_size:
        input_tdim = patch_size

    print("Spectrogram config:")
    print(f"  Frames/segment : {time_frames_per_segment} -> input_tdim: {input_tdim}")
    print(
        f"  Context length : {args.context_length} past + 1 current = {args.context_length+1} slots"
    )
    print(
        f"  Patches/slot   : {args.n_mels//patch_size} freq x {input_tdim//patch_size} time"
        f" = {(args.n_mels//patch_size)*(input_tdim//patch_size)}"
    )
    print(
        f"  Transformer tokens: "
        f"{(args.n_mels//patch_size)*(input_tdim//patch_size)*(args.context_length+1)} + 1 cls"
    )
    print(
        f"  Viz: every {args.viz_every_n_epochs} epochs, {args.viz_num_samples} samples"
    )

    model = LiveifyModel(
        input_fdim=args.n_mels,
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
        context_length=args.context_length,
    )

    lightning_module = LiveifyLightningModule(
        model=model,
        learning_rate=args.learning_rate,
        sample_rate=args.sample_rate,
        n_mels=args.n_mels,
        n_fft=args.n_fft,
        hop_length=hop_length,
        use_augmentation=args.use_augmentation,
        aug_freq_mask=args.aug_freq_mask,
        aug_time_mask=args.aug_time_mask,
        aug_noise_std=args.aug_noise_std,
        viz_every_n_epochs=args.viz_every_n_epochs,
        viz_num_samples=args.viz_num_samples,
        viz_log_audio=args.viz_log_audio,
        viz_save_local=args.viz_save_local,
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

    if args.logger == "wandb":
        logger = WandbLogger(
            project=args.wandb_project,
            name=args.wandb_run_name or None,
            save_dir=args.log_dir,
            log_model=False,
        )
    else:
        logger = TensorBoardLogger(save_dir=args.log_dir, name="liveify")

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="auto",
        devices=1,
        precision=args.precision,
        callbacks=[checkpoint_callback],
        logger=logger,
        log_every_n_steps=10,  # Reduced from 1 to cut wandb overhead
        gradient_clip_val=1,
        gradient_clip_algorithm="norm",
        accumulate_grad_batches=args.accumulate_grad_batches,
    )

    print("\n" + "=" * 50)
    print(
        f"Resuming from: {args.resume_from}"
        if args.resume_from
        else "Starting training..."
    )
    print("=" * 50 + "\n")

    trainer.fit(lightning_module, datamodule, ckpt_path=args.resume_from)

    print("\n" + "=" * 50)
    print("Training complete!")
    print(f"Best checkpoint: {checkpoint_callback.best_model_path}")
    print("=" * 50 + "\n")


# ===== Args ======


def parse_args():
    parser = argparse.ArgumentParser(description="Train Liveify model")

    # data
    parser.add_argument("--studio_dir", type=str, default="./datasetv2/studio")
    parser.add_argument("--live_dir", type=str, default="./datasetv2/live")
    parser.add_argument("--sample_rate", type=int, default=22050)
    parser.add_argument("--segment_duration", type=float, default=1.0)
    parser.add_argument("--context_length", type=int, default=8)
    parser.add_argument("--train_split", type=float, default=0.8)

    # training
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument(
        "--learning_rate", type=float, default=3e-4
    )  # was 1e-5 - increased for small dataset
    parser.add_argument("--max_epochs", type=int, default=-1)
    parser.add_argument("--patience", type=int, default=1000)
    parser.add_argument("--num_workers", type=int, default=10)
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)
    parser.add_argument(
        "--precision",
        type=str,
        default="bf16-mixed",
        choices=["32", "16", "bf16", "16-mixed", "bf16-mixed"],
    )

    # spectrogram
    parser.add_argument("--n_mels", type=int, default=128)
    parser.add_argument("--n_fft", type=int, default=2048)

    # model - REDUCED FOR SMALL DATASET
    parser.add_argument("--embed_dim", type=int, default=256)  # was 512
    parser.add_argument("--num_transformer_layers", type=int, default=2)  # was 4
    parser.add_argument("--num_heads", type=int, default=2)  # was 4
    parser.add_argument("--mlp_ratio", type=float, default=4.0)
    parser.add_argument("--dropout", type=float, default=0.1)  # was 0.3
    parser.add_argument("--attention_dropout", type=float, default=0.1)  # was 0.3
    parser.add_argument("--patch_size", type=int, default=16)

    # augmentation - LESS AGGRESSIVE FOR SMALL DATASET
    parser.add_argument("--use_augmentation", action="store_true", default=True)
    parser.add_argument(
        "--no_augmentation", dest="use_augmentation", action="store_false"
    )
    parser.add_argument(
        "--aug_freq_mask", type=int, default=10
    )  # was 20 - reduced masking
    parser.add_argument(
        "--aug_time_mask", type=int, default=20
    )  # was 40 - reduced masking
    parser.add_argument(
        "--aug_noise_std", type=float, default=0.02
    )  # was 0.1 - less noise

    # visualisation
    parser.add_argument(
        "--viz_every_n_epochs",
        type=int,
        default=1,
        help="Generate full visualisation panels every N epochs.",
    )
    parser.add_argument(
        "--viz_num_samples",
        type=int,
        default=4,
        help="Number of fixed validation examples to track and visualise.",
    )
    parser.add_argument(
        "--viz_log_audio",
        action="store_true",
        default=True,
        help="Upload Griffin-Lim audio to W&B so you can actually listen to outputs.",
    )
    parser.add_argument(
        "--no_viz_log_audio", dest="viz_log_audio", action="store_false"
    )
    parser.add_argument(
        "--viz_save_local",
        action="store_true",
        default=True,
        help="Save visualisation PNGs to ./spectrograms/epoch_XXXX/.",
    )
    parser.add_argument(
        "--no_viz_save_local", dest="viz_save_local", action="store_false"
    )

    # logging
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    parser.add_argument("--log_dir", type=str, default="./logs")
    parser.add_argument(
        "--logger",
        type=str,
        default="tensorboard",
        choices=["tensorboard", "wandb"],
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="liveify",
        help="W&B project name (--logger wandb only).",
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="W&B run display name (optional).",
    )
    parser.add_argument("--resume_from", type=str, default=None)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
