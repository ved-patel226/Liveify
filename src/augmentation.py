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
        # additional safeguards to avoid destroying the entire spec
        # masks will be at most this fraction of the corresponding dimension
        self.max_freq_mask_frac = 0.2
        self.max_time_mask_frac = 0.2
        # value used to fill masked regions (zero by default but can be mean/median)
        self.mask_value = 0.0

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

            # frequency masks (limit size relative to F)
            max_f = int(F * self.max_freq_mask_frac)
            for _ in range(self.num_freq_masks):
                if max_f <= 0:
                    break
                f = int(torch.rand(1).item() * min(self.freq_mask_param, max_f))
                if f == 0:
                    continue
                f0 = int(torch.rand(1).item() * (F - f))
                spec[b, :, f0 : f0 + f, :] = self.mask_value

            # time masks (limit size relative to T)
            max_t = int(T * self.max_time_mask_frac)
            for _ in range(self.num_time_masks):
                if max_t <= 0:
                    break
                t = int(torch.rand(1).item() * min(self.time_mask_param, max_t))
                if t == 0:
                    continue
                t0 = int(torch.rand(1).item() * max(1, T - t))
                spec[b, :, :, t0 : t0 + t] = self.mask_value

            if self.noise_std > 0:
                # scale noise by the mean magnitude of this spec to avoid overwhelming it
                mean_mag = spec[b].abs().mean().clamp(min=1e-6)
                noise = torch.randn_like(spec[b]) * self.noise_std * mean_mag
                spec[b] = spec[b] + noise

        # clamp to the original range in case noise pushed values outside
        spec = spec.clamp(-1.0, 1.0)

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
