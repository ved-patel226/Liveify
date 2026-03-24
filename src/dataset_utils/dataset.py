import os
import torch
import numpy as np
import librosa
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple, List
import pytorch_lightning as pl
from tqdm import tqdm
import whisper
import re
from difflib import SequenceMatcher
import pickle
import hashlib


def _align_pair_worker(args):
    """Module-level function for multiprocessing (must be pickleable)."""
    paths, pair_data, sr = args
    hop_length = 512

    studio_audio = pair_data["studio_audio"]
    live_audio = pair_data["live_audio"]

    # + 1e-9 to avoid DTW bitching
    studio_chroma = (
        librosa.feature.chroma_cqt(y=studio_audio, sr=sr, hop_length=hop_length) + 1e-9
    )
    live_chroma = (
        librosa.feature.chroma_cqt(y=live_audio, sr=sr, hop_length=hop_length) + 1e-9
    )

    _, wp = librosa.sequence.dtw(X=live_chroma, Y=studio_chroma, metric="cosine")

    studio_start_frame = wp[-1, 1]
    studio_start_sample = studio_start_frame * hop_length
    end_sample = studio_start_sample + len(live_audio)
    cropped_studio = studio_audio[studio_start_sample:end_sample]

    return paths, cropped_studio


class StudioLiveDataset(Dataset):
    version = "0.2.0"

    def __init__(
        self,
        studio_dir,
        live_dir,
        segment_duration=5.0,
        num_segments=5,
        lyric_match_threshold=0.5,
        context_length=16,
        forward_context_length=0,
        sr=22050,
        segment_overlap=0.5,
    ):
        """Dataset pairs `studio`/`live` audio files.

        Args:
            context_length: Number of past frames (before target)
            forward_context_length: Number of future frames from studio only (after target)
        """
        self.studio_dir = studio_dir
        self.live_dir = live_dir
        self.segment_duration = segment_duration
        self.num_segments = num_segments
        self.segment_samples = int(segment_duration * sr)
        self.lyric_match_threshold = lyric_match_threshold
        self.context_length = context_length
        self.forward_context_length = forward_context_length
        self.sr = sr
        self.segment_overlap = segment_overlap
        self.training = False  # Set to True during training for augmentation

        self.segment_hop = int(self.segment_samples * (1.0 - segment_overlap))
        self.segment_hop = max(self.segment_hop, 1)  # safety

        self.has_gpu = torch.cuda.is_available()

        studio_files = sorted(os.listdir(self.studio_dir))
        live_files = sorted(os.listdir(self.live_dir))

        self.pairs = [
            (os.path.join(self.studio_dir, sf), os.path.join(self.live_dir, lf))
            for sf, lf in zip(studio_files, live_files)
            if sf == lf
        ]

        unmatched_studio = set(studio_files) - set(live_files)
        unmatched_live = set(live_files) - set(studio_files)
        if unmatched_studio:
            print(f"Warning: Studio files with no pairs: {list(unmatched_studio)}")
        if unmatched_live:
            print(f"Warning: Live files with no pairs: {list(unmatched_live)}")

        # cache a flag so repeated setup calls (Lightning can call twice) do not re-align
        self._has_setup = False

        self.pairs_cache = {}
        self._get_local_cache_audio()
        self._align_cache_audio()

        if self.pairs_cache:
            min_samples = min(len(v["live_audio"]) for v in self.pairs_cache.values())
            total_context_slots = self.context_length + self.forward_context_length + 1
            min_needed = total_context_slots * self.segment_samples
            usable = min_samples - min_needed
            self._segments_per_song = max(1, usable // self.segment_hop + 1)
        else:
            self._segments_per_song = 0

    def _get_local_cache_audio(self):
        for studio_path, live_path in self.pairs:
            try:
                studio_audio, _ = librosa.load(studio_path, sr=self.sr, mono=True)
                live_audio, _ = librosa.load(live_path, sr=self.sr, mono=True)
                self.pairs_cache[(studio_path, live_path)] = {
                    "studio_audio": studio_audio,
                    "live_audio": live_audio,
                }
            except Exception as e:
                print(f"Error loading {studio_path} or {live_path}: {e}")
                continue

    def _align_cache_audio(self):
        """Align audio pairs using DTW. Can be parallelized for faster initialization."""
        from multiprocessing import Pool

        items = [
            (paths, pair_data, self.sr) for paths, pair_data in self.pairs_cache.items()
        ]

        # use 4 processes for DTW alignment (faster than serial)
        with Pool(processes=4) as pool:
            results = pool.imap_unordered(_align_pair_worker, items, chunksize=1)
            for paths, aligned_studio in tqdm(
                results, total=len(items), desc="Aligning audio pairs"
            ):
                self.pairs_cache[paths]["studio_audio"] = aligned_studio

    def _align_audio(self, studio_audio, live_audio, **kwargs):
        """Crop studio_audio to match live_audio using DTW on chroma features.
        No padding; returns cropped studio and original live audio.
        """
        hop_length = 512  # TODO: make this a parameter

        studio_chroma = librosa.feature.chroma_cqt(
            y=studio_audio, sr=self.sr, hop_length=hop_length
        )
        live_chroma = librosa.feature.chroma_cqt(
            y=live_audio, sr=self.sr, hop_length=hop_length
        )

        # dtw distance and path
        _, wp = librosa.sequence.dtw(X=live_chroma, Y=studio_chroma, metric="cosine")

        studio_start_frame = wp[-1, 1]
        studio_start_sample = studio_start_frame * hop_length
        end_sample = studio_start_sample + len(live_audio)
        cropped_studio = studio_audio[studio_start_sample:end_sample]

        return cropped_studio, live_audio

    def _calculate_text_similarity(self, text1, text2):
        if not text1 and not text2:
            return 1.0
        if not text1 or not text2:
            return 0.0
        return SequenceMatcher(None, text1, text2).ratio()

    def _generate_augmentation_params(self):
        """Generate random augmentation parameters for a single segment.
        Returns same params to apply to both studio and live for alignment.
        """
        if not self.training:
            return {"pitch_steps": 0, "time_rate": 1.0}

        pitch_steps = 0
        if np.random.rand() < 0.5:
            pitch_steps = np.random.uniform(-2, 2)

        time_rate = 1.0
        if np.random.rand() < 0.5:
            time_rate = np.random.uniform(0.95, 1.05)

        return {"pitch_steps": pitch_steps, "time_rate": time_rate}

    def _apply_augmentation(self, audio: np.ndarray, params: dict) -> np.ndarray:
        """Apply audio augmentation with specific parameters.
        Pitch shift and time stretch preserve alignment when same params used.
        """
        pitch_steps = params["pitch_steps"]
        time_rate = params["time_rate"]

        # Pitch shift ±2 semitones (changes timbre without breaking alignment)
        if pitch_steps != 0:
            audio = librosa.effects.pitch_shift(audio, sr=self.sr, n_steps=pitch_steps)

        # Speed change ±5% (subtle, preserves musical content)
        if time_rate != 1.0:
            audio = librosa.effects.time_stretch(audio, rate=time_rate)
            # Truncate or pad back to original length
            target_len = self.segment_samples
            if len(audio) > target_len:
                audio = audio[:target_len]
            else:
                audio = np.pad(audio, (0, target_len - len(audio)))

        return audio

    def __len__(self):
        return len(self.pairs) * self._segments_per_song

    def __getitem__(self, idx):
        song_idx = idx // self._segments_per_song
        seg_idx = idx % self._segments_per_song

        studio_path, live_path = self.pairs[song_idx]
        audio_dict = self.pairs_cache[(studio_path, live_path)]

        total_slots = self.context_length + self.forward_context_length + 1
        studio_out = np.zeros((total_slots, self.segment_samples))
        live_out = np.zeros((total_slots, self.segment_samples))

        # ═══ CHANGE: use segment_hop for start position ═══
        target_start = seg_idx * self.segment_hop

        # Fill backward context slots
        for i in range(self.context_length):
            slot_start = target_start - (self.context_length - i) * self.segment_hop
            if slot_start >= 0:
                s, e = slot_start, slot_start + self.segment_samples
                if e <= len(audio_dict["studio_audio"]):
                    studio_out[i] = audio_dict["studio_audio"][s:e]
                    live_out[i] = audio_dict["live_audio"][s:e]

        # Fill forward context (studio only)
        for fw in range(self.forward_context_length):
            slot_start = target_start + (1 + fw) * self.segment_hop
            s, e = slot_start, slot_start + self.segment_samples
            slot_idx = self.context_length + fw
            if e <= len(audio_dict["studio_audio"]):
                studio_out[slot_idx] = audio_dict["studio_audio"][s:e]

        # Target slot (last)
        target_slot = total_slots - 1
        s, e = target_start, target_start + self.segment_samples
        if e <= len(audio_dict["studio_audio"]):
            studio_out[target_slot] = audio_dict["studio_audio"][s:e]
            live_out[target_slot] = audio_dict["live_audio"][s:e]

        for slot in range(total_slots):
            params = self._generate_augmentation_params()
            studio_out[slot] = self._apply_augmentation(studio_out[slot], params)
            live_out[slot] = self._apply_augmentation(live_out[slot], params)

        return {
            "studio_audio": torch.tensor(studio_out, dtype=torch.float32),
            "live_audio": torch.tensor(live_out, dtype=torch.float32),
            "num_context": self.context_length,
            "id": live_path,
            "segment_idx": seg_idx,
            "cache_key": f"{live_path}::seg{seg_idx}",
        }


class StudioLiveDataModule(pl.LightningDataModule):
    def __init__(
        self,
        studio_dir: str,
        live_dir: str,
        batch_size: int = 8,
        sr: int = 22050,
        segment_duration: float = 5.0,
        num_segments: int = 5,
        lyric_match_threshold: float = 0.5,
        context_length: int = 16,
        forward_context_length: int = 0,
        train_split: float = 0.8,
        num_workers: int = 4,
        **dataset_kwargs,
    ):
        super().__init__()
        self.studio_dir = studio_dir
        self.live_dir = live_dir
        self.batch_size = batch_size
        self.sr = sr
        self.segment_duration = segment_duration
        self.num_segments = num_segments
        self.lyric_match_threshold = lyric_match_threshold
        self.context_length = context_length
        self.forward_context_length = forward_context_length
        self.train_split = train_split
        self.num_workers = num_workers
        self.segment_overlap = dataset_kwargs.get("segment_overlap", 0.75)
        self.dataset_kwargs = dataset_kwargs

    def setup(self, stage: Optional[str] = None):
        if getattr(self, "_has_setup", False):
            return
        full_dataset = StudioLiveDataset(
            studio_dir=self.studio_dir,
            live_dir=self.live_dir,
            segment_duration=self.segment_duration,
            num_segments=self.num_segments,
            lyric_match_threshold=self.lyric_match_threshold,
            context_length=self.context_length,
            forward_context_length=self.forward_context_length,
            sr=self.sr,
            segment_overlap=self.segment_overlap,
            **self.dataset_kwargs,
        )

        n_songs = len(full_dataset.pairs)
        n_train_songs = int(n_songs * self.train_split)

        # split song indices
        song_indices = list(range(n_songs))
        train_song_indices = set(song_indices[:n_train_songs])

        # create datasets based on song splits
        train_indices = [
            i
            for i in range(len(full_dataset))
            if (i // full_dataset._segments_per_song) in train_song_indices
        ]
        val_indices = [
            i
            for i in range(len(full_dataset))
            if (i // full_dataset._segments_per_song) not in train_song_indices
        ]

        # Enable augmentation for training dataset only
        full_dataset.training = True
        self.train_dataset = torch.utils.data.Subset(full_dataset, train_indices)

        # Disable augmentation for validation dataset
        full_dataset.training = False
        self.val_dataset = torch.utils.data.Subset(full_dataset, val_indices)

        n_train = len(train_indices)
        n_val = len(val_indices)
        print(
            f"Split by songs: {n_train_songs}/{n_songs} songs -> {n_train} train segments, {n_val} val segments"
        )
        self._has_setup = True

    def train_dataloader(self) -> DataLoader:
        effective_workers = (
            min(self.num_workers, 2)
            if len(self.train_dataset) < 100
            else self.num_workers
        )
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=effective_workers,
            persistent_workers=effective_workers > 0,
            pin_memory=True,
            prefetch_factor=2 if effective_workers > 0 else None,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,  # No shuffling in val, so no workers needed
            pin_memory=True,
        )


if __name__ == "__main__":
    dataset = StudioLiveDataset(
        studio_dir="./dataset/studio",
        live_dir="./dataset/live",
        sr=22050,
        segment_duration=0.5,
        context_length=16,
    )

    print(f"Dataset size: {len(dataset)}")

    if len(dataset) > 0:
        x, y = dataset[0]
        print(f"Studio (x): {x.shape}")
        print(f"Live (y): {y.shape}")
