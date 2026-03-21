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
    # Recreate the alignment logic here
    hop_length = 512

    studio_audio = pair_data["studio_audio"]
    live_audio = pair_data["live_audio"]

    studio_chroma = librosa.feature.chroma_cqt(
        y=studio_audio, sr=sr, hop_length=hop_length
    )
    live_chroma = librosa.feature.chroma_cqt(y=live_audio, sr=sr, hop_length=hop_length)

    _, wp = librosa.sequence.dtw(X=live_chroma, Y=studio_chroma, metric="cosine")

    studio_start_frame = wp[-1, 1]
    studio_start_sample = studio_start_frame * hop_length
    end_sample = studio_start_sample + len(live_audio)
    cropped_studio = studio_audio[studio_start_sample:end_sample]

    return paths, cropped_studio


class StudioLiveDataset(Dataset):
    __version__ = "0.1.0"

    def __init__(
        self,
        studio_dir,
        live_dir,
        segment_duration=5.0,
        num_segments=5,
        lyric_match_threshold=0.5,
        context_length=16,
        sr=22050,
    ):
        """Dataset pairs `studio`/`live` audio files."""
        self.studio_dir = studio_dir
        self.live_dir = live_dir
        self.segment_duration = segment_duration
        self.num_segments = num_segments
        self.segment_samples = int(segment_duration * sr)
        self.lyric_match_threshold = lyric_match_threshold
        self.context_length = context_length
        self.sr = sr

        self.has_gpu = torch.cuda.is_available()

        studio_files = sorted(os.listdir(self.studio_dir))
        live_files = sorted(os.listdir(self.live_dir))

        # keep pairs as list of tuples so we can use them as dict keys
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

        self.pairs_cache = {}
        self._get_local_cache_audio()
        self._align_cache_audio()

        if self.pairs_cache:
            min_samples = min(len(v["live_audio"]) for v in self.pairs_cache.values())
            self._segments_per_song = max(1, min_samples // self.segment_samples)
        else:
            self._segments_per_song = 0

        print(self.pairs_cache)

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

        # Prepare arguments for worker function
        items = [
            (paths, pair_data, self.sr) for paths, pair_data in self.pairs_cache.items()
        ]

        # Use 4 processes for DTW alignment (faster than serial)
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

    def __len__(self):
        return len(self.pairs) * self._segments_per_song

    def __getitem__(self, idx):
        song_idx = idx // self._segments_per_song
        target_segment = idx % self._segments_per_song

        studio_path, live_path = self.pairs[song_idx]
        audio_dict = self.pairs_cache[(studio_path, live_path)]

        context_start = max(0, target_segment - self.context_length)
        num_context = (
            target_segment - context_start
        )  # how many real context segments exist

        # pre-allocate with zeros
        studio_out = np.zeros((self.context_length + 1, self.segment_samples))
        live_out = np.zeros((self.context_length + 1, self.segment_samples))

        for i, seg_idx in enumerate(range(context_start, target_segment + 1)):
            s = seg_idx * self.segment_samples
            e = s + self.segment_samples
            # right-align: target always at [-1], context fills in from the right
            slot = (self.context_length - num_context) + i
            studio_out[slot] = audio_dict["studio_audio"][s:e]
            live_out[slot] = audio_dict["live_audio"][s:e]

        return {
            "studio_audio": torch.tensor(
                studio_out, dtype=torch.float32
            ),  # (context_length+1, segment_samples)
            "live_audio": torch.tensor(live_out, dtype=torch.float32),
            "num_context": num_context,
            "id": live_path,
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
        self.train_split = train_split
        self.num_workers = num_workers
        self.dataset_kwargs = dataset_kwargs

    def setup(self, stage: Optional[str] = None):
        full_dataset = StudioLiveDataset(
            studio_dir=self.studio_dir,
            live_dir=self.live_dir,
            segment_duration=self.segment_duration,
            num_segments=self.num_segments,
            lyric_match_threshold=self.lyric_match_threshold,
            context_length=self.context_length,
            sr=self.sr,
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

        self.train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
        self.val_dataset = torch.utils.data.Subset(full_dataset, val_indices)

        n_train = len(train_indices)
        n_val = len(val_indices)
        print(
            f"Split by songs: {n_train_songs}/{n_songs} songs -> {n_train} train segments, {n_val} val segments"
        )

    def train_dataloader(self) -> DataLoader:
        # Reduce workers if dataset is small to avoid overhead
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
