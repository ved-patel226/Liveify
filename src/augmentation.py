import torch
import torch.nn as nn
import numpy as np


class SpectrogramAugmentation(nn.Module):
    """
    Data augmentation for spectrograms during training.
    Implements SpecAugment-style masking + Gaussian noise.
    """

    def __init__(
        self,
        freq_mask_param: int = 20,
        time_mask_param: int = 40,
        num_freq_masks: int = 2,
        num_time_masks: int = 2,
        noise_std: float = 0.01,
        p: float = 0.5,
    ):
        """
        Args:
            freq_mask_param: Maximum width of frequency mask
            time_mask_param: Maximum width of time mask
            num_freq_masks: Number of frequency masks to apply
            num_time_masks: Number of time masks to apply
            noise_std: Standard deviation of Gaussian noise to add
            p: Probability of applying augmentation to each sample
        """
        super().__init__()
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.num_freq_masks = num_freq_masks
        self.num_time_masks = num_time_masks
        self.noise_std = noise_std
        self.p = p

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        """
        Apply augmentation to spectrogram.

        Args:
            spec: (batch, channels, freq, time) or (channels, freq, time)

        Returns:
            Augmented spectrogram with same shape
        """
        if not self.training:
            return spec

        batch_mode = spec.dim() == 4
        if not batch_mode:
            spec = spec.unsqueeze(0)

        B, C, F, T = spec.shape
        spec = spec.clone()

        for b in range(B):
            if torch.rand(1).item() > self.p:
                continue

            for _ in range(self.num_freq_masks):
                f = int(torch.rand(1).item() * self.freq_mask_param)
                f0 = int(torch.rand(1).item() * (F - f))
                spec[b, :, f0 : f0 + f, :] = 0

            for _ in range(self.num_time_masks):
                t = int(torch.rand(1).item() * self.time_mask_param)
                t0 = int(torch.rand(1).item() * max(1, T - t))
                spec[b, :, :, t0 : t0 + t] = 0

            if self.noise_std > 0:
                noise = torch.randn_like(spec[b]) * self.noise_std
                spec[b] = spec[b] + noise

        if not batch_mode:
            spec = spec.squeeze(0)

        return spec


class ComposedAugmentation(nn.Module):
    """Compose multiple augmentations."""

    def __init__(self, *augmentations):
        super().__init__()
        self.augmentations = nn.ModuleList(augmentations)

    def forward(self, x):
        for aug in self.augmentations:
            x = aug(x)
        return x
