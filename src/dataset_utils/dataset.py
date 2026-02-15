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


def lyric_similarity_score(text1: str, text2: str) -> float:
    """Simple similarity score between two text strings."""
    text1 = re.sub(r"[^\w\s]", "", text1.lower())
    text2 = re.sub(r"[^\w\s]", "", text2.lower())
    return SequenceMatcher(None, text1, text2).ratio()


class StudioLiveDataset(Dataset):
    def __init__(
        self,
        studio_dir: str,
        live_dir: str,
        sr: int = 22050,
        segment_duration: float = 0.5,
        context_length: int = 16,
        development_mode: bool = False,
        min_lyric_similarity: float = 0.3,
        n_mels: int = 256,
        cache_dir: str = "./cache",
    ):
        """
        Dataset for paired studio/live audio aligned via lyrics.

        Args:
            studio_dir: Path to studio recordings
            live_dir: Path to live recordings
            sr: Sample rate
            segment_duration: Duration of each segment in seconds
            context_length: Number of consecutive segments per sample
            development_mode: If True, only load first pair
            min_lyric_similarity: Minimum similarity to keep segment
            n_mels: Number of mel bins
            cache_dir: Cache directory
        """
        self.studio_dir = studio_dir
        self.live_dir = live_dir
        self.sr = sr
        self.segment_length = int(segment_duration * sr)
        self.context_length = context_length
        self.min_lyric_similarity = min_lyric_similarity
        self.n_mels = n_mels
        self.cache_dir = cache_dir
        self.development_mode = development_mode

        self.whisper_model = None
        self.pairs = self._load_or_create_segments()

    def _get_cache_path(self) -> str:
        """Generate cache file path."""
        config = f"{self.studio_dir}_{self.live_dir}_{self.sr}_{self.segment_length}_{self.min_lyric_similarity}_{self.development_mode}_{self.n_mels}_v2"
        cache_key = hashlib.md5(config.encode()).hexdigest()
        os.makedirs(self.cache_dir, exist_ok=True)
        return os.path.join(self.cache_dir, f"segments_{cache_key}.pkl")

    def _load_or_create_segments(self) -> List[dict]:
        """Load from cache or create new segments."""
        cache_path = self._get_cache_path()

        if os.path.exists(cache_path):
            try:
                print(f"Loading from cache: {cache_path}")
                with open(cache_path, "rb") as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"Cache load failed: {e}")

        segments = self._create_segments()

        try:
            with open(cache_path, "wb") as f:
                pickle.dump(segments, f)
            print(f"Saved to cache: {cache_path}")
        except Exception as e:
            print(f"Cache save failed: {e}")

        return segments

    def _audio_to_mel(self, audio: np.ndarray) -> np.ndarray:
        """Convert audio to normalized mel spectrogram."""
        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sr,
            n_fft=2048,
            hop_length=512,
            n_mels=self.n_mels,
            fmin=20,
            fmax=8000,
        )
        mel_db = librosa.power_to_db(mel, ref=1.0)
        mel_db = np.clip(mel_db, -80.0, 20.0)
        mel_norm = 2.0 * (mel_db + 80.0) / 100.0 - 1.0
        return mel_norm

    def _create_segments(self) -> List[dict]:
        """Create all segments from audio pairs."""
        if self.whisper_model is None:
            print("Loading Whisper model...")
            self.whisper_model = whisper.load_model("small")

        studio_files = sorted(
            [f for f in os.listdir(self.studio_dir) if f.endswith((".mp3", ".wav"))]
        )
        live_files = sorted(
            [f for f in os.listdir(self.live_dir) if f.endswith((".mp3", ".wav"))]
        )

        pairs = []
        for sf in studio_files:
            for lf in live_files:
                if os.path.splitext(sf)[0].lower() in os.path.splitext(lf)[0].lower():
                    pairs.append((sf, lf))

        if self.development_mode:
            pairs = pairs[:1]

        all_segments = []
        print(f"Processing {len(pairs)} pairs...")

        for studio_file, live_file in tqdm(pairs):
            segments = self._process_pair(studio_file, live_file)
            all_segments.extend(segments)

        print(f"Created {len(all_segments)} segments")
        return all_segments

    def _process_pair(self, studio_file: str, live_file: str) -> List[dict]:
        """Process a single audio pair - maintains temporal order."""
        studio_path = os.path.join(self.studio_dir, studio_file)
        live_path = os.path.join(self.live_dir, live_file)

        studio_audio, _ = librosa.load(studio_path, sr=self.sr)
        live_audio, _ = librosa.load(live_path, sr=self.sr)

        try:
            studio_trans = self.whisper_model.transcribe(
                studio_path, language="en", fp16=False, verbose=False
            )
            live_trans = self.whisper_model.transcribe(
                live_path, language="en", fp16=False, verbose=False
            )
        except Exception as e:
            print(f"Transcription failed for {studio_file}: {e}")
            return []

        segment_matches = []
        for studio_seg in studio_trans.get("segments", []):
            studio_text = studio_seg.get("text", "").strip()
            best_score = 0.0
            best_live_seg = None

            for live_seg in live_trans.get("segments", []):
                live_text = live_seg.get("text", "").strip()
                score = lyric_similarity_score(studio_text, live_text)
                if score > best_score:
                    best_score = score
                    best_live_seg = live_seg

            if best_live_seg and best_score >= self.min_lyric_similarity:
                segment_matches.append(
                    {
                        "studio_start": studio_seg["start"],
                        "studio_end": studio_seg["end"],
                        "live_start": best_live_seg["start"],
                        "live_end": best_live_seg["end"],
                        "similarity": best_score,
                    }
                )

        segments = []
        for match in segment_matches:
            studio_clip = self._extract_clip(
                studio_audio, match["studio_start"], match["studio_end"]
            )
            live_clip = self._extract_clip(
                live_audio, match["live_start"], match["live_end"]
            )

            min_len = min(len(studio_clip), len(live_clip))
            studio_clip = studio_clip[:min_len]
            live_clip = live_clip[:min_len]

            num_segments = min_len // self.segment_length

            for i in range(num_segments):
                start = i * self.segment_length
                end = start + self.segment_length

                studio_seg_audio = studio_clip[start:end]
                live_seg_audio = live_clip[start:end]

                seg_dict = {
                    "studio_spec": self._audio_to_mel(studio_seg_audio),
                    "live_spec": self._audio_to_mel(live_seg_audio),
                    "studio_name": studio_file,
                    "live_name": live_file,
                    "similarity": match["similarity"],
                    "timestamp": match["studio_start"]
                    + (i * self.segment_length / self.sr),
                    "studio_audio": studio_seg_audio,
                    "live_audio": live_seg_audio,
                }

                segments.append(seg_dict)

        return segments

    def _extract_clip(self, audio: np.ndarray, start: float, end: float) -> np.ndarray:
        """Extract audio clip from timestamps."""
        start_sample = int(start * self.sr)
        end_sample = int(end * self.sr)
        return audio[max(0, start_sample) : min(len(audio), end_sample)]

    def _build_windows(self) -> List[List[Optional[int]]]:
        """Build context windows with padding - maintains temporal continuity."""
        if self.context_length <= 1:
            return [[i] for i in range(len(self.pairs))]

        groups = {}
        for i, pair in enumerate(self.pairs):
            key = (pair["studio_name"], pair["live_name"])
            groups.setdefault(key, []).append(i)

        for key in groups:
            groups[key].sort(key=lambda i: self.pairs[i]["timestamp"])

        windows = []
        for song_key, indices in groups.items():
            for i in range(len(indices)):
                window = []
                padding_needed = max(0, self.context_length - 1 - i)
                window.extend([None] * padding_needed)
                start = max(0, i - (self.context_length - 1))
                window.extend(indices[start : i + 1])
                windows.append(window)

        num_padded = sum(None in w for w in windows)
        print(f"Built {len(windows)} windows ({num_padded} with padding)")
        return windows

    def __len__(self) -> int:
        if not hasattr(self, "_windows"):
            self._windows = self._build_windows()
        return len(self._windows)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a window of spectrograms."""
        if not hasattr(self, "_windows"):
            self._windows = self._build_windows()

        window = self._windows[idx]

        if self.context_length <= 1:
            pair = self.pairs[window[0]]
            x = torch.from_numpy(pair["studio_spec"]).float()
            y = torch.from_numpy(pair["live_spec"]).float()
            return x, y

        studio_specs = []
        live_specs = []

        silent_shape = self.pairs[0]["studio_spec"].shape
        silent_spec = np.full(silent_shape, -1.0)

        for pair_idx in window:
            if pair_idx is None:
                studio_specs.append(silent_spec)
                live_specs.append(silent_spec)
            else:
                studio_specs.append(self.pairs[pair_idx]["studio_spec"])
                live_specs.append(self.pairs[pair_idx]["live_spec"])

        x = torch.from_numpy(np.stack(studio_specs)).float()
        y = torch.from_numpy(np.stack(live_specs)).float()
        return x, y


class StudioLiveDataModule(pl.LightningDataModule):
    def __init__(
        self,
        studio_dir: str,
        live_dir: str,
        batch_size: int = 8,
        sr: int = 22050,
        segment_duration: float = 0.5,
        context_length: int = 16,
        train_split: float = 0.8,
        num_workers: int = 4,
        development_mode: bool = False,
        **dataset_kwargs,
    ):
        super().__init__()
        self.studio_dir = studio_dir
        self.live_dir = live_dir
        self.batch_size = batch_size
        self.sr = sr
        self.segment_duration = segment_duration
        self.context_length = context_length
        self.train_split = train_split
        self.num_workers = num_workers
        self.development_mode = development_mode
        self.dataset_kwargs = dataset_kwargs

    def setup(self, stage: Optional[str] = None):
        full_dataset = StudioLiveDataset(
            self.studio_dir,
            self.live_dir,
            sr=self.sr,
            segment_duration=self.segment_duration,
            context_length=self.context_length,
            development_mode=self.development_mode,
            **self.dataset_kwargs,
        )

        n_total = len(full_dataset)
        n_train = int(n_total * self.train_split)
        n_val = n_total - n_train

        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            full_dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42)
        )

        print(f"Split: {n_train} train, {n_val} val")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            pin_memory=True,
        )


if __name__ == "__main__":
    dataset = StudioLiveDataset(
        studio_dir="./dataset/studio",
        live_dir="./dataset/live",
        sr=22050,
        segment_duration=0.5,
        context_length=16,
        development_mode=True,
    )

    print(f"Dataset size: {len(dataset)}")

    if len(dataset) > 0:
        x, y = dataset[0]
        print(f"Studio (x): {x.shape}")
        print(f"Live (y): {y.shape}")
