# Liveify

A deep learning model that transforms studio audio into live, performance audio. (similar to the Weekend's Live at SoFi Stadium album)

## Architecture

### Model Overview

Liveify uses a Vision Transformer-inspired architecture that operates on mel-spectrogram patches rather than raw audio. The model treats spectrograms as 2D images and learns to transform studio frequency-time patterns into live performance characteristics.

### Model Components

**Patch Embedding**

- Converts mel-spectrograms into non-overlapping patches (default 16×16)
- Projects patches into embedding space via 2D convolution
- Input: `(batch, 1, 128_mels, time_frames)` → Output: `(batch, num_patches, embed_dim)`

**Positional Encoding**

- Learnable 2D positional embeddings for frequency and time dimensions
- Separate embeddings for frequency and time axes, concatenated
- Includes a learnable CLS token (not used in reconstruction)

**Transformer Encoder Layers (Pre-GRU)**

- Multi-head self-attention with scaled dot-product attention (Fast Attention)
- MLP with GELU activation (4× hidden dimension expansion)
- Default: 4 layers before GRU

**Bidirectional GRU Context Module**

- Captures sequential dependencies across patch embeddings
- Processes the full temporal sequence with residual connection
- 2-layer bidirectional GRU with layer normalization

**Transformer Encoder Layers (Post-GRU)**

- Additional transformer layers after GRU processing
- Same architecture as pre-GRU layers
- Default: 4 layers after GRU (8 total transformer layers)

**Patch Reconstruction**

- Projects patch embeddings back to spatial patches
- Unpatchifies into full spectrogram: `(batch, num_patches, embed_dim)` → `(batch, 1, 128, time`

### Data Processing Pipeline

1. **Audio → Mel-Spectrogram**: 128 mel bands, 2048 FFT, 512 hop length
2. **Log Scale**: `power_to_db` for perceptual scaling ([-80, 0] dB)
3. **Normalization**: Scale to [-1, 1] to match Tanh output: `(dB / 40) + 1`
4. **Context Windows**: Concatenate consecutive segments along time axis
5. **Data Augmentation** (training only):
   - **SpecAugment**: Random time and frequency masking
   - **Gaussian Noise**: Small random noise addition
   - Applied to 50% of training samples
6. **Patching**: Divide into 16×16 patches for transformer processing

### Installation

### Prerequisites

- Python 3.9+
- CUDA 11.8+

### Setup# Clone repository

```bash
rcd /mnt/Fedora2/code/Liveify

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install tensorboard torchinfo  # For training utilities
```

## Dataset Preparation

The Liveify dataset is available on Hugging Face and includes pre-aligned studio/live audio pairs.

### Quick Start

```bash
# Download dataset from Hugging Face
from huggingface_hub import snapshot_download

dataset_path = snapshot_download(
    repo_id="ved-patel226/Liveify_Dataset",
    repo_type="dataset",
    local_dir="./dataset"
)
```

Or clone with Git LFS:

```bash
git clone https://huggingface.co/datasets/ved-patel226/Liveify_Dataset
cd Liveify_Dataset
```

### Directory Structure

Once downloaded, your dataset will look like:

```
dataset/
├── studio/
│   ├── song1.mp3
│   ├── song2.wav
│   └── ...
└── live/
    ├── song1.mp3
    ├── song2.wav
    └── ...
```

The dataset includes:

- **Matched pairs** by filename
- **Pre-analyzed alignment** via Whisper transcription
- **Lyric-based matching** with configurable thresholds
- **Automatic caching** for fast subsequent loads

### Adding Your Own Pairs

To add custom studio/live pairs:

1. Place files in `dataset/studio/` and `dataset/live/` with matching names
2. First training run will automatically:
   - Transcribe both versions with Whisper
   - Align segments by lyric similarity
   - Cache processed segments
   - Normalize audio to [-1, 1]

**Note**: First run is slow (~2-5 min per pair) due to Whisper transcription. Subsequent runs load from cache instantly.

See [dataset/README.md](dataset/README.md) for detailed dataset documentation.

## Training

### Basic Usage

```bash
python src/train.py \
  --studio_dir ./dataset/studio \
  --live_dir ./dataset/live \
  --batch_size 4 \
  --segment_duration 0.5 \
  --context_length 64 \
  --max_epochs 100 \
  --learning_rate 1e-6 \
  --logger wandb
```

**Note**: With `segment_duration=0.5` and `context_length=64`, each training sample covers 32 seconds of audio (64 × 0.5s), concatenated along the time axis.

### Command-Line Arguments

**Dataset:**
Duration of each audio segment in seconds (default: 0.5)

- `--context_length`: Number of consecutive segments concatenated for temporal context (default: 64)
- `--train_split`: Train/validation split ratio (default: 0.8)
- `--development_mode`: Use only first song pair for fast iteration

**Training:**

- `--batch_size`: Batch size (default: 8)
- `--learning_rate`: Initial learning rate (default: 1e-6)
- `--max_epochs`: Maximum training epochs (default: 100)
- `--patience`: Early stopping patience (default: 10)
- `--num_workers`: DataLoader workers (default: 10)
- `--accumulate_grad_batches`: Gradient accumulation steps (default: 1)
- `--precision`: Training precision - "32", "16", "bf16", "16-mixed", "bf16-mixed" (default: "bf16-mixed")

**Model Architecture:**

- `--embed_dim`: Transformer embedding dimension (default: 512)
- `--num_transformer_layers`: Total transformer layers (split pre/post GRU) (default: 8)
- `--num_heads`: Number of attention heads (default: 8)
- `--mlp_ratio`: MLP hidden dimension ratio (default: 4.0)
- `--gru_layers`: Number of bidirectional GRU layers (default: 2)
- `--dropout`: Dropout rate (default: 0.1)
- `--attention_dropout`: Attention dropout rate (default: 0.1)
- `--patch_size`: Patch size for vision transformer (default: 16)

**Data Augmentation:**

- `--use_augmentation`: Enable data augmentation (default: enabled)
- `--no_augmentation`: Disable data augmentation
- `--aug_freq_mask`: Maximum frequency mask width for SpecAugment (default: 20)
- `--aug_time_mask`: Maximum time mask width for SpecAugment (default: 40)
- `--aug_noise_std`: Standard deviation of Gaussian noise (default: 0.01)

**Checkpoints & Logging:**

- `--checkpoint_dir`: Directory for model checkpoints (default: `./checkpoints`)
- `--log_dir`: Directory for logs (default: `./logs`)
  **With TensorBoard (default):**

```bash
tensorboard --logdir ./logs
```

Then open `http://localhost:6006/` in your browser.

**With Weights & Biases:**

```bash
python src/train.py --logger wandb
```

The dashboard shows:

- Training/validation loss curves
- Learning rate schedule
- Gradient norms
- Per-epoch metric
  tensorboard --logdir ./logs
  The model architecture is automatically configured based on `segment_duration` and `context_length`:

```python
# Example with segment_duration=0.5s, context_length=64, patch_size=16
time_frames_per_segment = int((22050 * 0.5) / 512)  # ~22 frames per segment
total_time_frames = 22 * 64  # 1408 frames total (32 seconds)
input_tdim = (1408 // 16) * 16  # 1408 (aligned to patch size)

model = LiveifyModel(
    input_fdim=128,              # mel bands
    input_tdim=1408,             # time frames (context_length × frames_per_segment)
    patch_size=(16, 16),         # 16×16 patches
    embed_dim=512,               # embedding dimension
    num_transformer_layers=8,    # 4 pre-GRU + 4 post-GRU
    num_heads=8,                 # attention heads
    mlp_ratio=4.0,               # MLP expansion ratio
    gru_layers=2,                # bidirectional GRU layers
    dropout=0.1,
    attention_dropout=0.1,
)

# This creates: 8 freq patches × 88 time patches = 704 total patches
```

Reduce memory usage by adjusting these parameters:

```bash
python src/train.py \
  --batch_size 2 \
  --context_length 32 \
  --patch_size 16 \
  --embed_dim 384 \
  --num_transformer_layers 6
```

Or use gradient accumulation to simulate larger batches:

```bash
python src/train.py --batch_size 2 --accumulate_grad_batches 4  # Effective batch size: 8
```

### Slow Dataset Loading

First run processes and caches segments (includes Whisper transcription and spectrogram generation). Subsequent runs load from cache (~100x faster).

Delete cache to force reprocessing after changing:

- `segment_duration`
- `context_length`
- `sample_rate`
- Audio files in dataset

```bash
rm -rf ./src/cache/ ./cache/
```

### High Training Loss (100+)

If loss remains in hundreds, the normalization may be incorrect. Ensure:

1. Cache is cleared after updating dataset code
2. Dataset returns spectrograms in [-1, 1] range
3. Model uses Tanh activation (outputs [-1, 1])

The current implementation normalizes spectrograms: `(power_to_db / 40) + 1` which maps [-80, 0] dB → [-1, 1].

### Overfitting

If validation loss is much higher than training loss:

1. **Data augmentation** (enabled by default): SpecAugment masking + Gaussian noise
2. **Increase dropout**: `--dropout 0.2 --attention_dropout 0.2`
3. **Reduce model capacity**: `--embed_dim 384 --num_transformer_layers 6`
4. **More training data**: Add additional studio/live pairs to dataset
5. **Disable augmentation for testing**: Use `--no_augmentation` to verify augmentation isn't too aggressive

- `context_length`
- `sample_rate`
- Audio files in dataset

```bash
rm -rf ./src/cache/ ./cache/
```

### Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Submit a pull request with clear descriptions

## Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper) for automatic speech recognition
- [PyTorch Lightning](https://www.pytorchlightning.ai/) for training utilities
- [librosa](https://librosa.org/) for audio processing
