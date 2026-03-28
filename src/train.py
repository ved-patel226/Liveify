import torch

torch.set_float32_matmul_precision("high")

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
        segment_overlap=args.segment_overlap,
    )

    print("Encodec latent model config:")
    print(
        f"  Context length : {args.context_length} past + {args.forward_context_length} future + 1 current = "
        f"{args.context_length + args.forward_context_length + 1} slots"
    )

    print(f"  Latent dim     : 128 (Encodec encoder output)")
    print(
        f"  Model          : Cross-attention transformer, layers={args.latent_layers}"
    )

    model = EncodecLatentModel(
        latent_dim=128,
        context_length=args.context_length,
        forward_context_length=args.forward_context_length,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.latent_layers,
        ff_mult=args.ff_mult,
        dropout=args.dropout,
        drop_path=args.drop_path,
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
        log_every_n_steps=10,
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
    parser.add_argument("--studio_dir", type=str, default="../datasetv2/studio")
    parser.add_argument("--live_dir", type=str, default="../datasetv2/live")
    parser.add_argument("--sample_rate", type=int, default=48000)
    parser.add_argument("--segment_duration", type=float, default=0.5)
    parser.add_argument("--segment_overlap", type=float, default=0.75)
    parser.add_argument("--context_length", type=int, default=12)
    parser.add_argument(
        "--forward_context_length",
        type=int,
        default=24,
        help="Number of future frames from studio to include as context (live remains zero-padded).",
    )
    parser.add_argument("--train_split", type=float, default=0.85)
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
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--ff_mult", type=int, default=2)
    parser.add_argument("--drop_path", type=float, default=0.1)
    parser.add_argument("--dropout", type=float, default=0.3)
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
