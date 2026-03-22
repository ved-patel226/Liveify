import torch

torch.set_float32_matmul_precision("high")

import torchaudio
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pathlib import Path
from typing import List, Optional, Tuple
import argparse
import io
import hashlib

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")

import librosa
import numpy as np

from encodec import EncodecModel

from models import (
    EncodecLatentModel,
    EncodecLatentLightningModule,
)
from dataset_utils.dataset import StudioLiveDataModule


_SPEC_KW = dict(aspect="auto", origin="lower", cmap="magma", vmin=-1, vmax=1)
_DIFF_KW = dict(aspect="auto", origin="lower", cmap="RdBu_r", vmin=-1, vmax=1)


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


def train(args=None):
    if args is None:
        args = parse_args()

    pl.seed_everything(42)

    encodec_sample_rate = 24000

    datamodule = StudioLiveDataModule(
        studio_dir=args.studio_dir,
        live_dir=args.live_dir,
        batch_size=args.batch_size,
        sr=encodec_sample_rate,
        segment_duration=args.segment_duration,
        context_length=args.context_length,
        forward_context_length=args.forward_context_length,
        train_split=args.train_split,
        num_workers=args.num_workers,
    )

    print("Encodec latent model config:")
    print(
        f"  Context length : {args.context_length} past + 1 current = {args.context_length+1} slots"
    )
    print(f"  Latent dim     : 128 (Encodec encoder output)")
    print(
        f"  Model          : Cross-attention transformer, layers={args.latent_layers}"
    )

    model = EncodecLatentModel(
        latent_dim=128,
        context_length=args.context_length,
        forward_context_length=args.forward_context_length,
        num_layers=args.latent_layers,
        dropout=args.dropout,
    )

    lightning_module = EncodecLatentLightningModule(
        model=model,
        learning_rate=args.learning_rate,
        sample_rate=encodec_sample_rate,
        encodec_bandwidth=args.encodec_bandwidth,
        encodec_sample_rate=encodec_sample_rate,
        forward_context_length=args.forward_context_length,
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

    datamodule.setup()

    n_train = len(datamodule.train_dataset)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ratio = n_params / n_train
    print(f"Training samples: {n_train}")
    print(f"Trainable params: {n_params:,}")
    print(f"Params/sample ratio: {ratio:.0f}")

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
    parser.add_argument("--sample_rate", type=int, default=48000)
    parser.add_argument("--segment_duration", type=float, default=1.0)
    parser.add_argument("--context_length", type=int, default=8)
    parser.add_argument(
        "--forward_context_length",
        type=int,
        default=0,
        help="Number of future frames from studio to include as context (live remains zero-padded).",
    )
    parser.add_argument("--train_split", type=float, default=0.8)
    parser.add_argument(
        "--encodec_bandwidth",
        type=float,
        default=6.0,
        help="Target bandwidth kbps for Encodec encoder (controls compression).",
    )

    # training
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
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

    # model
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--latent_layers",
        type=int,
        default=4,
        help="Number of cross-attention blocks for Encodec latent model.",
    )

    # logging
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    parser.add_argument("--log_dir", type=str, default="./logs")
    parser.add_argument(
        "--logger",
        type=str,
        default="wandb",
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
