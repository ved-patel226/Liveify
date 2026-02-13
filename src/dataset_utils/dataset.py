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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import hashlib


def preprocess_lyrics(text):
    """Clean and normalize lyrics for comparison."""
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\b(oh|ah|yeah|no)\b", "", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def calculate_sequence_similarity(text1, text2):
    """Calculate similarity using sequence matching."""
    return SequenceMatcher(None, text1, text2).ratio()


def calculate_tfidf_similarity(text1, text2):
    """Calculate similarity using TF-IDF vectors."""
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
    try:
        tfidf_matrix = vectorizer.fit_transform([text1, text2])
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return similarity
    except ValueError:
        return 0.0


def extract_key_phrases(text, num_phrases=10):
    """Extract key phrases from lyrics."""
    words = text.split()
    phrases = []

    for n in range(2, 5):
        for i in range(len(words) - n + 1):
            phrase = " ".join(words[i : i + n])
            if len(phrase) > 5:
                phrases.append(phrase)

    return phrases[:num_phrases]


def phrase_overlap_similarity(text1, text2):
    """Calculate similarity based on overlapping key phrases."""
    phrases1 = set(extract_key_phrases(text1))
    phrases2 = set(extract_key_phrases(text2))

    if not phrases1 or not phrases2:
        return 0.0

    intersection = phrases1.intersection(phrases2)
    union = phrases1.union(phrases2)

    return len(intersection) / len(union) if union else 0.0


def lyric_similarity_score(studio_lyrics, live_lyrics):
    """
    Calculate comprehensive similarity score between studio and live lyrics.
    Returns a score between 0 and 1, where 1 is identical.
    """
    studio_clean = preprocess_lyrics(studio_lyrics)
    live_clean = preprocess_lyrics(live_lyrics)

    sequence_sim = calculate_sequence_similarity(studio_clean, live_clean)
    tfidf_sim = calculate_tfidf_similarity(studio_clean, live_clean)
    phrase_sim = phrase_overlap_similarity(studio_clean, live_clean)

    final_score = 0.4 * sequence_sim + 0.4 * tfidf_sim + 0.2 * phrase_sim

    return final_score


class StudioLiveDataset(Dataset):
    def __init__(
        self,
        studio_dir: str,
        live_dir: str,
        sr: int = 22050,
        max_offset_sec: float = 150.0,
        segment_length: Optional[int] = None,
        segment_duration: Optional[float] = None,
        context_length: int = 1,
        transform=None,
        development_mode: bool = True,
        min_lyric_similarity: float = 0.3,
        cache_dir: Optional[str] = "./cache",
    ):
        """
        Args:
            studio_dir: Path to studio recordings directory
            live_dir: Path to live recordings directory
            sr: Sample rate for audio loading
            max_offset_sec: Maximum offset for alignment in seconds
            segment_length: Fixed segment length in samples (None = use full aligned segment)
            segment_duration: Fixed segment duration in seconds (overrides segment_length if provided)
            context_length: Number of consecutive segments to return per sample (for LSTM temporal context)
            transform: Optional transform to apply to audio
            development_mode: If True, only load the first song pair for faster iteration
            min_lyric_similarity: Minimum lyric similarity score (0-1) to include segment (uses Whisper)
            cache_dir: Directory to store cached segments (None to disable caching)
        """
        self.studio_dir = studio_dir
        self.live_dir = live_dir
        self.sr = sr
        self.max_offset = int(max_offset_sec * sr)
        self.cache_dir = cache_dir
        self.context_length = context_length

        # segment_duration -> segment_length if provided
        if segment_duration is not None:
            self.segment_length = int(segment_duration * sr)
        else:
            self.segment_length = segment_length

        min_alignment_samples = int(5.0 * sr)
        if (
            self.segment_length is not None
            and self.segment_length < min_alignment_samples
        ):
            self.alignment_length = min_alignment_samples
        else:
            self.alignment_length = self.segment_length

        self.transform = transform
        self.development_mode = development_mode
        self.min_lyric_similarity = min_lyric_similarity

        self.whisper_model = None

        self.pairs = self._find_and_align_pairs()

    def _get_cache_key(self) -> str:
        """Generate a unique cache key based on dataset configuration."""
        config_str = f"{self.studio_dir}_{self.live_dir}_{self.sr}_{self.segment_length}_{self.min_lyric_similarity}_{self.development_mode}_{self.context_length}_spec_db-80_20_v5"
        return hashlib.md5(config_str.encode()).hexdigest()

    def _get_cache_path(self) -> Optional[str]:
        """Get the path to the cache file."""
        if self.cache_dir is None:
            return None
        os.makedirs(self.cache_dir, exist_ok=True)
        cache_key = self._get_cache_key()
        return os.path.join(self.cache_dir, f"segments_{cache_key}.pkl")

    def _load_from_cache(self) -> Optional[List[dict]]:
        """Load segments from cache if available."""
        cache_path = self._get_cache_path()
        if cache_path and os.path.exists(cache_path):
            try:
                print(f"Loading segments from cache: {cache_path}")
                with open(cache_path, "rb") as f:
                    pairs = pickle.load(f)
                print(f"Loaded {len(pairs)} segments from cache")
                return pairs
            except Exception as e:
                print(f"Warning: Failed to load cache: {e}")
                return None
        return None

    def _save_to_cache(self, pairs: List[dict]):
        """Save segments to cache."""
        cache_path = self._get_cache_path()
        if cache_path:
            try:
                print(f"Saving {len(pairs)} segments to cache: {cache_path}")
                with open(cache_path, "wb") as f:
                    pickle.dump(pairs, f)
                print("Cache saved successfully")
            except Exception as e:
                print(f"Warning: Failed to save cache: {e}")

    def _audio_to_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """Convert audio waveform to mel-spectrogram, normalized to [-1, 1].

        Args:
            audio: Audio waveform array (should NOT be pre-normalized)

        Returns:
            Normalized mel-spectrogram array with shape (n_mels, time_frames),
            values in [-1, 1] (maps from power_to_db range [-80, 20] dB).
        """
        # mel-spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sr,
            n_fft=2048,
            hop_length=512,
            n_mels=256,
            fmin=20,
            fmax=8000,
        )

        mel_spec_db = librosa.power_to_db(mel_spec, ref=1.0)

        db_min, db_max = -80.0, 20.0
        mel_spec_db = np.clip(mel_spec_db, db_min, db_max)
        mel_spec_norm = 2.0 * (mel_spec_db - db_min) / (db_max - db_min) - 1.0

        return mel_spec_norm

    def _load_audio(self, path: str) -> np.ndarray:
        y, _ = librosa.load(path, sr=self.sr)
        return y

    def _find_lyric_matches(
        self, studio_transcription: dict, live_transcription: dict
    ) -> List[dict]:
        """Find matching segments based on lyric similarity using timestamps."""
        studio_segments = studio_transcription.get("segments", [])
        live_segments = live_transcription.get("segments", [])

        matches = []

        for studio_seg in studio_segments:
            studio_text = studio_seg.get("text", "").strip()

            best_score = 0.0
            best_live_seg = None

            for live_seg in live_segments:
                live_text = live_seg.get("text", "").strip()

                score = lyric_similarity_score(studio_text, live_text)

                if score > best_score:
                    best_score = score
                    best_live_seg = live_seg

            if best_live_seg and best_score >= self.min_lyric_similarity:
                matches.append(
                    {
                        "studio_start": studio_seg.get("start", 0),
                        "studio_end": studio_seg.get("end", 0),
                        "live_start": best_live_seg.get("start", 0),
                        "live_end": best_live_seg.get("end", 0),
                        "studio_text": studio_text,
                        "live_text": best_live_seg.get("text", "").strip(),
                        "similarity": best_score,
                    }
                )

        return matches

    def _extract_audio_clip(
        self, audio: np.ndarray, start_time: float, end_time: float
    ) -> np.ndarray:
        """Extract audio clip based on time range."""
        start_sample = int(start_time * self.sr)
        end_sample = int(end_time * self.sr)

        start_sample = max(0, start_sample)
        end_sample = min(len(audio), end_sample)

        return audio[start_sample:end_sample]

    def _ensure_whisper_loaded(self):
        """Load Whisper model if not already loaded."""
        if self.whisper_model is None:
            print("Loading Whisper model for lyric alignment...")
            self.whisper_model = whisper.load_model("small")

    def _process_pair(self, studio_name: str, live_name: str) -> List[dict]:
        """Process a single audio pair: transcribe, find lyric matches, extract audio segments."""
        studio_path = os.path.join(self.studio_dir, studio_name)
        live_path = os.path.join(self.live_dir, live_name)

        studio_audio = self._load_audio(studio_path)
        live_audio = self._load_audio(live_path)

        self._ensure_whisper_loaded()

        studio_transcription = None
        live_transcription = None
        try:
            studio_transcription = self.whisper_model.transcribe(
                studio_path, language="en", fp16=False, verbose=False
            )
            live_transcription = self.whisper_model.transcribe(
                live_path, language="en", fp16=False, verbose=False
            )
        except Exception as e:
            error_msg = str(e)
            studio_duration = len(studio_audio) / self.sr
            live_duration = len(live_audio) / self.sr

            if "key.size" in error_msg and "value.size" in error_msg:
                print(
                    f"Warning: Whisper transcription failed for {studio_name} (dimension mismatch)"
                )
                print(
                    f"  Studio audio: {len(studio_audio)} samples ({studio_duration:.1f}s)"
                )
                print(f"  Live audio: {len(live_audio)} samples ({live_duration:.1f}s)")
                print(f"  This may indicate corrupted audio or an unexpected format.")
                print(f"  Raw error: {error_msg[:100]}...")
            else:
                print(
                    f"Warning: Whisper transcription failed for {studio_name}: {error_msg}"
                )
                print(
                    f"  Studio duration: {studio_duration:.1f}s, Live duration: {live_duration:.1f}s"
                )

            return [], 0, 0.0

        if not studio_transcription or not live_transcription:
            return [], 0, 0.0

        lyric_matches = self._find_lyric_matches(
            studio_transcription, live_transcription
        )

        segments = []

        for i, match in enumerate(lyric_matches):
            studio_seg = self._extract_audio_clip(
                studio_audio, match["studio_start"], match["studio_end"]
            )
            live_seg = self._extract_audio_clip(
                live_audio, match["live_start"], match["live_end"]
            )

            studio_real_len = len(studio_seg)
            live_real_len = len(live_seg)

            if self.segment_length is not None and self.segment_length < max(
                studio_real_len, live_real_len
            ):
                max_real_samples = min(studio_real_len, live_real_len)
                num_sub = max_real_samples // self.segment_length

                for j in range(num_sub):
                    start = j * self.segment_length
                    end = start + self.segment_length
                    sub_studio = studio_seg[start:end]
                    sub_live = live_seg[start:end]

                    studio_rms = np.sqrt(np.mean(sub_studio**2))
                    live_rms = np.sqrt(np.mean(sub_live**2))
                    if studio_rms < 0.01 or live_rms < 0.01:
                        continue

                    # do NOT peak-normalize per segment
                    studio_spec = self._audio_to_spectrogram(sub_studio)
                    live_spec = self._audio_to_spectrogram(sub_live)

                    seg_duration = self.segment_length / self.sr
                    studio_abs_start = match["studio_start"] + j * seg_duration
                    live_abs_start = match["live_start"] + j * seg_duration

                    segments.append(
                        {
                            "studio": sub_studio,
                            "live": sub_live,
                            "studio_spec": studio_spec,
                            "live_spec": live_spec,
                            "studio_name": studio_name,
                            "live_name": live_name,
                            "correlation": 0.0,
                            "lyric_similarity": match["similarity"],
                            "offset": match["live_start"] - match["studio_start"],
                            "segment_idx": i,
                            "sub_segment_idx": j,
                            "studio_text": match["studio_text"],
                            "live_text": match["live_text"],
                            "studio_start_sec": studio_abs_start,
                            "live_start_sec": live_abs_start,
                        }
                    )
            else:
                # no chopping needed - segment_length == alignment_length
                if self.segment_length is not None:
                    studio_seg = studio_seg[: self.segment_length]
                    live_seg = live_seg[: self.segment_length]

                studio_rms = np.sqrt(np.mean(studio_seg**2))
                live_rms = np.sqrt(np.mean(live_seg**2))
                if studio_rms < 0.01 or live_rms < 0.01:
                    continue

                # Do NOT peak-normalize — use raw audio for spectrogram
                studio_spec = self._audio_to_spectrogram(studio_seg)
                live_spec = self._audio_to_spectrogram(live_seg)

                segments.append(
                    {
                        "studio": studio_seg,
                        "live": live_seg,
                        "studio_spec": studio_spec,
                        "live_spec": live_spec,
                        "studio_name": studio_name,
                        "live_name": live_name,
                        "correlation": 0.0,
                        "lyric_similarity": match["similarity"],
                        "offset": match["live_start"] - match["studio_start"],
                        "segment_idx": i,
                        "studio_text": match["studio_text"],
                        "live_text": match["live_text"],
                        "studio_start_sec": match["studio_start"],
                        "live_start_sec": match["live_start"],
                    }
                )

        if segments:
            offsets = [seg["offset"] for seg in segments]
            median_offset = np.median(offsets)
        else:
            median_offset = 0.0

        return segments, median_offset, 0.0

    def _find_and_align_pairs(self) -> List[dict]:
        """Find matching files, align them, and split into segments using parallel processing."""
        cached_pairs = self._load_from_cache()
        if cached_pairs is not None:
            return cached_pairs

        self._ensure_whisper_loaded()

        pairs = []

        studio_files = sorted(
            [f for f in os.listdir(self.studio_dir) if f.endswith((".mp3", ".wav"))]
        )
        live_files = sorted(
            [f for f in os.listdir(self.live_dir) if f.endswith((".mp3", ".wav"))]
        )

        matching_pairs = []
        for studio_name in studio_files:
            for live_name in live_files:
                studio_base = os.path.splitext(studio_name)[0].lower()
                live_base = os.path.splitext(live_name)[0].lower()

                if studio_base in live_base or live_base in studio_base:
                    matching_pairs.append((studio_name, live_name))

        if not matching_pairs:
            print("No matching pairs found!")
            return pairs

        if self.development_mode:
            print(f"Development mode: Loading only first pair...")
            matching_pairs = matching_pairs[:1]
        else:
            print(f"Finding and aligning {len(matching_pairs)} pairs...")

        with tqdm(total=len(matching_pairs), desc="Processing pairs") as pbar:
            for studio_name, live_name in matching_pairs:
                try:
                    segments, offset, corr = self._process_pair(studio_name, live_name)
                    pairs.extend(segments)

                    num_segs = len(segments)
                    seg_info = f"{num_segs} segment{'s' if num_segs != 1 else ''}"
                    if self.segment_length is not None:
                        seg_info += f" of {self.segment_length/self.sr:.1f}s"

                    avg_similarity = (
                        np.mean([s["lyric_similarity"] for s in segments])
                        if segments
                        else 0.0
                    )
                    pbar.set_postfix_str(
                        f"{studio_name[:20]}... → offset={offset:.1f}s, sim={avg_similarity:.3f}, {seg_info}"
                    )
                    pbar.update(1)
                except Exception as e:
                    print(f"\n  Error processing {studio_name} ↔ {live_name}: {e}")
                    pbar.update(1)

        print(f"\nTotal segments: {len(pairs)}")

        self._save_to_cache(pairs)

        return pairs

    def _build_context_indices(self):
        """
        Build index for context_length windows of consecutive segments.
        Groups segments by song, sorts by absolute timestamp, and only creates
        windows from segments that are truly temporally contiguous (no gaps
        from skipped silent segments or cross-lyric-match boundaries).
        """
        if self.context_length <= 1:
            return [(i, 1) for i in range(len(self.pairs))]

        seg_duration = self.segment_length / self.sr if self.segment_length else None
        # seg_duration + leftover + whisper_gap (~0.8-2.0s for 0.5s segments).
        # jumps to completely different song sections.
        continuity_tolerance = seg_duration * 4.0 if seg_duration else None

        groups = {}
        for i, pair in enumerate(self.pairs):
            key = (pair["studio_name"], pair["live_name"])
            if key not in groups:
                groups[key] = []
            groups[key].append(i)

        for key in groups:
            groups[key].sort(
                key=lambda idx: self.pairs[idx].get("studio_start_sec", 0.0)
            )

        windows = []
        for key, indices in groups.items():
            if len(indices) < self.context_length:
                continue

            contiguous_runs = []
            current_run = [indices[0]]

            for k in range(1, len(indices)):
                prev_pair = self.pairs[indices[k - 1]]
                curr_pair = self.pairs[indices[k]]

                prev_studio_start = prev_pair.get("studio_start_sec", 0.0)
                curr_studio_start = curr_pair.get("studio_start_sec", 0.0)

                time_gap = curr_studio_start - prev_studio_start
                is_contiguous = (
                    continuity_tolerance is not None
                    and 0 < time_gap <= continuity_tolerance
                )

                if is_contiguous:
                    current_run.append(indices[k])
                else:
                    contiguous_runs.append(current_run)
                    current_run = [indices[k]]

            contiguous_runs.append(current_run)

            run_lengths = [len(r) for r in contiguous_runs]
            print(
                f"  Song {key[0][:25]}: {len(indices)} segments → {len(contiguous_runs)} contiguous runs (lengths: {run_lengths[:10]}{'...' if len(run_lengths) > 10 else ''})"
            )
            for run in contiguous_runs:
                if len(run) >= self.context_length:
                    for start in range(len(run) - self.context_length + 1):
                        window_indices = run[start : start + self.context_length]
                        windows.append(window_indices)

        print(
            f"Context windows: {len(windows)} (context_length={self.context_length}, tolerance={continuity_tolerance:.2f}s)"
        )
        return windows

    def __len__(self) -> int:
        if not hasattr(self, "_context_windows"):
            self._context_windows = self._build_context_indices()
        return len(self._context_windows)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            x: Studio audio (input) - shape (context_length, segment_length) or (segment_length,)
            y: Live audio (target) - same shape as x
        """
        if not hasattr(self, "_context_windows"):
            self._context_windows = self._build_context_indices()

        window = self._context_windows[idx]

        if self.context_length <= 1:
            if isinstance(window, tuple):
                pair_idx = window[0]
            else:
                pair_idx = window[0]

            pair = self.pairs[pair_idx]

            studio_spec = pair["studio_spec"]
            live_spec = pair["live_spec"]

            x = torch.from_numpy(studio_spec).float()
            y = torch.from_numpy(live_spec).float()

            if self.transform is not None:
                x = self.transform(x)
                y = self.transform(y)

            return x, y

        studio_specs = []
        live_specs = []

        for pair_idx in window:
            pair = self.pairs[pair_idx]
            studio_specs.append(pair["studio_spec"])
            live_specs.append(pair["live_spec"])

        # (context_length, n_mels, time_frames)
        x = torch.from_numpy(np.stack(studio_specs)).float()
        y = torch.from_numpy(np.stack(live_specs)).float()

        if self.transform is not None:
            x = self.transform(x)
            y = self.transform(y)

        return x, y

    def get_pair_info(self, idx: int) -> dict:
        """Get metadata about a segment."""
        pair = self.pairs[idx]
        return {
            "studio_name": pair["studio_name"],
            "live_name": pair["live_name"],
            "correlation": pair["correlation"],
            "lyric_similarity": pair.get("lyric_similarity", 1.0),
            "offset_sec": pair["offset"] / self.sr,
            "duration": len(pair["studio"]) / self.sr,
            "segment_idx": pair.get("segment_idx", 0),
        }


def collate_variable_length(
    batch: List[Tuple[torch.Tensor, torch.Tensor]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Custom collate function that pads tensors to max length in batch."""
    studios, lives = zip(*batch)

    max_len = max(max(s.shape[0] for s in studios), max(l.shape[0] for l in lives))

    studios_padded = torch.stack(
        [torch.nn.functional.pad(s, (0, max_len - s.shape[0])) for s in studios]
    )
    lives_padded = torch.stack(
        [torch.nn.functional.pad(l, (0, max_len - l.shape[0])) for l in lives]
    )

    return studios_padded, lives_padded


class StudioLiveDataModule(pl.LightningDataModule):
    def __init__(
        self,
        studio_dir: str,
        live_dir: str,
        batch_size: int = 8,
        sr: int = 22050,
        segment_length: Optional[int] = None,
        segment_duration: Optional[float] = None,
        context_length: int = 1,
        train_split: float = 0.8,
        persistent_workers: bool = True,
        num_workers: int = 4,
        development_mode: bool = False,
        **dataset_kwargs,
    ):
        """
        Args:
            studio_dir: Path to studio recordings
            live_dir: Path to live recordings
            batch_size: Batch size for dataloaders
            sr: Sample rate
            segment_length: Fixed segment length in samples
            segment_duration: Fixed segment duration in seconds (overrides segment_length if provided)
            context_length: Number of consecutive segments per sample (for LSTM temporal context)
            train_split: Fraction of data for training (rest for validation)
            num_workers: Number of workers for dataloaders
            development_mode: If True, only load the first song pair for faster iteration
            **dataset_kwargs: Additional kwargs for StudioLiveDataset
        """
        super().__init__()
        self.studio_dir = studio_dir
        self.live_dir = live_dir
        self.batch_size = batch_size
        self.sr = sr
        self.context_length = context_length

        # segment_duration -> segment_length if provided
        if segment_duration is not None:
            self.segment_length = int(segment_duration * sr)
        else:
            self.segment_length = segment_length

        # for sub-5s segments, alignment happens at 5s then gets chopped
        min_alignment_samples = int(5.0 * sr)
        if (
            self.segment_length is not None
            and self.segment_length < min_alignment_samples
        ):
            self.alignment_length = min_alignment_samples
        else:
            self.alignment_length = self.segment_length

        self.train_split = train_split
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers
        self.development_mode = development_mode
        self.dataset_kwargs = dataset_kwargs

        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage: Optional[str] = None):
        """Setup train and validation datasets."""
        full_dataset = StudioLiveDataset(
            self.studio_dir,
            self.live_dir,
            sr=self.sr,
            segment_length=self.segment_length,
            context_length=self.context_length,
            development_mode=self.development_mode,
            **self.dataset_kwargs,
        )

        n_total = len(full_dataset)
        n_train = int(n_total * self.train_split)
        n_val = n_total - n_train

        if n_total > 0 and n_train == 0:
            n_train = 1
            n_val = n_total - 1

        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            full_dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42)
        )

        print(f"Dataset split: {n_train} train, {n_val} val")

    def train_dataloader(self) -> DataLoader:
        collate_fn = collate_variable_length if self.segment_length is None else None
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=(
                self.persistent_workers if self.num_workers > 0 else False
            ),
            pin_memory=True,
            collate_fn=collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        collate_fn = collate_variable_length if self.segment_length is None else None
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=(
                self.persistent_workers if self.num_workers > 0 else False
            ),
            pin_memory=True,
            collate_fn=collate_fn,
        )


if __name__ == "__main__":
    dataset = StudioLiveDataset(
        studio_dir="./dataset/studio",
        live_dir="./dataset/live",
        sr=22050,
        segment_duration=0.5,  # 500ms segments (aligned at 5s, then chopped)
        context_length=16,  # 16 consecutive 500ms frames
        development_mode=True,
        min_lyric_similarity=0.3,
    )

    print(f"\nDataset size: {len(dataset)}")

    if len(dataset) > 0:
        x, y = dataset[0]
        print(f"\nSample:")
        print(f"  Studio (x): {x.shape}, dtype={x.dtype}")
        print(f"  Live (y): {y.shape}, dtype={y.dtype}")

    dm = StudioLiveDataModule(
        studio_dir="./dataset/studio",
        live_dir="./dataset/live",
        batch_size=4,
        segment_duration=0.02,
        context_length=16,
        development_mode=False,
    )

    dm.setup()

    train_loader = dm.train_dataloader()
    batch_x, batch_y = next(iter(train_loader))
    print(f"\nBatch:")
    print(f"  Studio (x): {batch_x.shape}")
    print(f"  Live (y): {batch_y.shape}")
