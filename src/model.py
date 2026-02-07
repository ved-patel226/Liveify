import torch
import torch.nn as nn
import pytorch_lightning as pl
from pathlib import Path
import argparse

from dataset.dataset import StudioLiveDataModule


class LiveifyModel(torch.nn.Module):
    def __init__(
        self,
        input_sr=110250,
        output_sr=110250,
        hidden_channels=128,
    ):
        super(LiveifyModel, self).__init__()

        self.input_sr = input_sr
        self.output_sr = output_sr

        # ===== Encoder =====
        self.encoder = nn.Sequential(
            nn.Conv1d(1, hidden_channels, kernel_size=15, padding=7),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Conv1d(hidden_channels, hidden_channels * 2, kernel_size=15, padding=7),
            nn.BatchNorm1d(hidden_channels * 2),
            nn.ReLU(),
        )

        # ===== Processor =====
        # TODO: add attention, transformers later
        self.processor = nn.Sequential(
            nn.Conv1d(
                hidden_channels * 2, hidden_channels * 2, kernel_size=15, padding=7
            ),
            nn.BatchNorm1d(hidden_channels * 2),
            nn.ReLU(),
            nn.Conv1d(
                hidden_channels * 2, hidden_channels * 2, kernel_size=15, padding=7
            ),
            nn.BatchNorm1d(hidden_channels * 2),
            nn.ReLU(),
        )

        # ===== Decoder =====
        self.decoder = nn.Sequential(
            nn.Conv1d(hidden_channels * 2, hidden_channels, kernel_size=15, padding=7),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Conv1d(hidden_channels, 1, kernel_size=15, padding=7),
            nn.Tanh(),  #  [-1, 1] range for audio
        )

    def forward(self, x):
        # (batch_size, sequence_length) -> (batch_size, 1, sequence_length)
        x = x.unsqueeze(1)

        encoded = self.encoder(x)
        processed = self.processor(encoded)
        output = self.decoder(processed)

        # (batch_size, 1, sequence_length) -> (batch_size, sequence_length)
        output = output.squeeze(1)

        return output  # (batch_size, sequence_length)
