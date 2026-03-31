import os
import copy
import torch
import numpy as np
import librosa
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple, List
import pytorch_lightning as pl
from tqdm import tqdm
import re
from difflib import SequenceMatcher
from multiprocessing import Pool


# ─────────────────────────────────────────────────────────────
# DTW alignment worker (must be top-level for pickling)
# ─────────────────────────────────────────────────────────────
def _align_pair_worker(args):
    paths, pair_data, sr = args
    hop_length = 512
    studio_audio = pair_data["studio_audio"]
    live_audio = pair_data["live_audio"]
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


# ─────────────────────────────────────────────────────────────
# Original audio dataset (unchanged)
# ─────────────────────────────────────────────────────────────
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
        segment_overlap=0.875,
    ):
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
        self.training = False

        self.segment_hop = int(self.segment_samples * (1.0 - segment_overlap))
        self.segment_hop = max(self.segment_hop, 1)

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

    def _align_cache_audio(self):
        items = [
            (paths, pair_data, self.sr) for paths, pair_data in self.pairs_cache.items()
        ]
        with Pool(processes=10) as pool:
            results = pool.imap_unordered(_align_pair_worker, items, chunksize=1)
            for paths, aligned_studio in tqdm(
                results, total=len(items), desc="Aligning audio pairs"
            ):
                self.pairs_cache[paths]["studio_audio"] = aligned_studio

    def _generate_augmentation_params(self):
        if not self.training:
            return {"pitch_steps": 0, "time_rate": 1.0}
        pitch_steps = 0
        if np.random.rand() < 0.5:
            pitch_steps = np.random.uniform(-2, 2)
        time_rate = 1.0
        if np.random.rand() < 0.5:
            time_rate = np.random.uniform(0.95, 1.05)
        return {"pitch_steps": pitch_steps, "time_rate": time_rate}

    def _apply_augmentation(self, audio, params):
        pitch_steps = params["pitch_steps"]
        time_rate = params["time_rate"]
        if pitch_steps != 0:
            audio = librosa.effects.pitch_shift(audio, sr=self.sr, n_steps=pitch_steps)
        if time_rate != 1.0:
            audio = librosa.effects.time_stretch(audio, rate=time_rate)
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

        target_start = seg_idx * self.segment_hop

        for i in range(self.context_length):
            s = target_start - (self.context_length - i) * self.segment_hop
            e = s + self.segment_samples
            if s >= 0 and e <= len(audio_dict["studio_audio"]):
                studio_out[i] = audio_dict["studio_audio"][s:e]
                live_out[i] = audio_dict["live_audio"][s:e]

        for fw in range(self.forward_context_length):
            slot_idx = self.context_length + fw
            s = target_start + (fw + 1) * self.segment_hop
            e = s + self.segment_samples
            if e <= len(audio_dict["studio_audio"]):
                studio_out[slot_idx] = audio_dict["studio_audio"][s:e]

        target_slot = total_slots - 1
        s, e = target_start, target_start + self.segment_samples
        if e <= len(audio_dict["studio_audio"]):
            studio_out[target_slot] = audio_dict["studio_audio"][s:e]
            live_out[target_slot] = audio_dict["live_audio"][s:e]

        if self.training:
            aug_params = self._generate_augmentation_params()
            for slot in range(total_slots):
                if np.any(studio_out[slot]):
                    studio_out[slot] = self._apply_augmentation(
                        studio_out[slot], aug_params
                    )
                if np.any(live_out[slot]):
                    live_out[slot] = self._apply_augmentation(
                        live_out[slot], aug_params
                    )

        return {
            "studio_audio": torch.tensor(studio_out, dtype=torch.float32),
            "live_audio": torch.tensor(live_out, dtype=torch.float32),
            "num_context": self.context_length,
            "id": live_path,
            "segment_idx": seg_idx,
            "cache_key": f"{live_path}::seg{seg_idx}",
        }


# ─────────────────────────────────────────────────────────────
# NEW: Precomputed-latent dataset
# ─────────────────────────────────────────────────────────────
class PrecomputedLatentDataset(Dataset):
    """Returns precomputed Encodec latent vectors.

    Each song's audio was pre-encoded on a regular hop grid:
        grid position g  ↔  audio[g·hop : g·hop + seg_samples]

    __getitem__ assembles context / forward-context / target slots by
    indexing into these grids — identical slot logic to StudioLiveDataset,
    but zero encoder cost.
    """

    def __init__(
        self,
        pairs: list,
        studio_grids: list,  # [song_i] → Tensor(n_grid_i, C, T_latent)
        live_grids: list,
        context_length: int,
        forward_context_length: int,
        segments_per_song: int,
    ):
        self.pairs = pairs
        self.studio_grids = studio_grids
        self.live_grids = live_grids
        self.context_length = context_length
        self.forward_context_length = forward_context_length
        self._segments_per_song = segments_per_song

    def __len__(self):
        return len(self.pairs) * self._segments_per_song

    def __getitem__(self, idx):
        song_idx = idx // self._segments_per_song
        seg_idx = idx % self._segments_per_song

        total_slots = self.context_length + self.forward_context_length + 1
        sg = self.studio_grids[song_idx]  # (n_grid, C, T)
        lg = self.live_grids[song_idx]
        n_grid, C, T = sg.shape

        s_out = sg.new_zeros(total_slots, C, T)
        l_out = lg.new_zeros(total_slots, C, T)

        tgt = seg_idx  # target's grid index

        # ── past context (both studio & live) ──
        for i in range(self.context_length):
            gp = tgt - (self.context_length - i)
            if 0 <= gp < n_grid:
                s_out[i] = sg[gp]
                l_out[i] = lg[gp]

        # ── forward context (studio only — live stays zero) ──
        for fw in range(self.forward_context_length):
            gp = tgt + fw + 1
            if 0 <= gp < n_grid:
                s_out[self.context_length + fw] = sg[gp]

        # ── target slot (last) ──
        if 0 <= tgt < n_grid:
            s_out[-1] = sg[tgt]
            l_out[-1] = lg[tgt]

        return {
            "studio_latents": s_out,
            "live_latents": l_out,
            "cache_key": f"{self.pairs[song_idx][1]}::seg{seg_idx}",
        }


# ─────────────────────────────────────────────────────────────
# DataModule (with precomputation support)
# ─────────────────────────────────────────────────────────────
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
        self.segment_overlap = dataset_kwargs.pop("segment_overlap", 0.75)
        self.dataset_kwargs = dataset_kwargs
        self._has_setup = False

    def setup(self, stage=None):
        if self._has_setup:
            return

        common_kwargs = dict(
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

        full_dataset = StudioLiveDataset(**common_kwargs)

        n_songs = len(full_dataset.pairs)
        n_train_songs = int(n_songs * self.train_split)
        train_song_indices = set(range(n_train_songs))

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

        train_ds = copy.copy(full_dataset)
        train_ds.training = True
        val_ds = copy.copy(full_dataset)
        val_ds.training = False

        self.train_dataset = torch.utils.data.Subset(train_ds, train_indices)
        self.val_dataset = torch.utils.data.Subset(val_ds, val_indices)

        print(
            f"Split: {n_train_songs}/{n_songs} songs → "
            f"{len(train_indices)} train, {len(val_indices)} val"
        )
        self._has_setup = True

    # ─────────────────────────────────────────────────────────
    # NEW: one-shot Encodec precomputation
    # ─────────────────────────────────────────────────────────
    def precompute_encodec_latents(
        self,
        encodec_model,
        encodec_sr: int = 24000,
        device: str = "cuda",
        encode_batch_size: int = 256,
    ):
        """Encode every hop-grid segment with Encodec **once**, then swap
        the audio-returning datasets for lightweight latent-returning ones.

        After this call the DataLoaders yield dicts with keys
        ``studio_latents`` / ``live_latents`` of shape ``(S, C, T)``
        instead of raw waveforms.
        """
        import torchaudio  # local import — only needed here

        assert self._has_setup, "call .setup() before .precompute_encodec_latents()"

        base_ds: StudioLiveDataset = self.train_dataset.dataset

        encodec_model = encodec_model.to(device)
        encodec_model.eval()

        studio_grids: list[torch.Tensor] = []
        live_grids: list[torch.Tensor] = []

        for sp, lp in tqdm(base_ds.pairs, desc="Pre-encoding with Encodec"):
            ad = base_ds.pairs_cache[(sp, lp)]

            sg = self._encode_audio_grid(
                ad["studio_audio"],
                base_ds.segment_samples,
                base_ds.segment_hop,
                encodec_model,
                encodec_sr,
                device,
                encode_batch_size,
            )
            lg = self._encode_audio_grid(
                ad["live_audio"],
                base_ds.segment_samples,
                base_ds.segment_hop,
                encodec_model,
                encodec_sr,
                device,
                encode_batch_size,
            )

            # studio/live can differ by a frame after alignment
            min_g = min(sg.shape[0], lg.shape[0])
            studio_grids.append(sg[:min_g])
            live_grids.append(lg[:min_g])

        base_ds.pairs_cache = {}

        lat_ds = PrecomputedLatentDataset(
            pairs=base_ds.pairs,
            studio_grids=studio_grids,
            live_grids=live_grids,
            context_length=base_ds.context_length,
            forward_context_length=base_ds.forward_context_length,
            segments_per_song=base_ds._segments_per_song,
        )

        train_idx = self.train_dataset.indices
        val_idx = self.val_dataset.indices
        self.train_dataset = torch.utils.data.Subset(lat_ds, train_idx)
        self.val_dataset = torch.utils.data.Subset(lat_ds, val_idx)

        total_gb = (
            sum(g.nelement() * g.element_size() for g in studio_grids + live_grids)
            / 1e9
        )
        print(
            f"✓ Precomputed latents for {len(studio_grids)} songs  |  "
            f"RAM ≈ {total_gb:.2f} GB  |  "
            f"grid sizes {[g.shape[0] for g in studio_grids]}"
        )

    def _encode_audio_grid(
        self,
        audio: np.ndarray,
        segment_samples: int,
        segment_hop: int,
        encodec_model,
        encodec_sr: int,
        device: str,
        batch_size: int,
    ) -> torch.Tensor:
        """Encode a full song on its hop grid → ``(n_grid, C, T_latent)``."""
        import torchaudio

        n_grid = max(1, (len(audio) - segment_samples) // segment_hop + 1)

        segments: list[np.ndarray] = []
        for g in range(n_grid):
            s = g * segment_hop
            e = s + segment_samples
            if e <= len(audio):
                segments.append(audio[s:e])
            else:
                seg = np.zeros(segment_samples, dtype=audio.dtype)
                seg[: len(audio) - s] = audio[s:]
                segments.append(seg)

        all_lats: list[torch.Tensor] = []
        for i in range(0, len(segments), batch_size):
            chunk = np.stack(segments[i : i + batch_size])
            wav = torch.from_numpy(chunk).float().unsqueeze(1)  # (B, 1, L)

            if self.sr != encodec_sr:
                wav = torchaudio.functional.resample(wav, self.sr, encodec_sr)

            wav = wav / wav.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
            wav = wav.to(device)

            with torch.no_grad():
                lat = encodec_model.encoder(wav)  # (B, C, T)

            all_lats.append(lat.cpu())

        return torch.cat(all_lats, dim=0)  # (n_grid, C, T)

    # ─────────────────────────────────────────────────────────
    # DataLoaders (unchanged)
    # ─────────────────────────────────────────────────────────
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
            num_workers=0,
            pin_memory=True,
        )
