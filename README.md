# Liveify

A deep learning model that transforms studio audio into live, performance audio. (similar to the Weekend's Live at SoFi Stadium album)

## Architecture

### Model Components

**Encoder**

- Convolutional encoder
- Progressively downsamples audio while increasing channel dimensions
- 256x total downsampling

**Transformer**

- Multi-head self-attention blocks
- Positional embeddings
- 8 heads, 4 layers, 384 hidden dimensions

**Decoder**

- Symmetric transposed convolutions matching the encoder
- Tanh activation for output normalization to [-1, 1]
- Residual connection from input to output

**Loss Function**

- **Time-Domain Loss**: L1 loss between predicted and target audio
- **Spectral Loss**: Multi-scale STFT magnitude and log-magnitude losses

## Installation

### Prerequisites

- Python 3.9+
- CUDA 11.8+ (optional, for GPU acceleration)

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
  --batch_size 8 \
  --segment_duration 4.0 \
  --max_epochs 100 \
  --learning_rate 1e-4
```

### Command-Line Arguments

**Dataset:**

- `--studio_dir`: Path to studio recordings (default: `./dataset/studio`)
- `--live_dir`: Path to live recordings (default: `./dataset/live`)
- `--sample_rate`: Audio sample rate in Hz (default: 22050)
- `--segment_duration`: Segment length in seconds (default: 4.0)
- `--train_split`: Train/validation split ratio (default: 0.8)
- `--development_mode`: Use only first song pair for fast iteration

**Training:**

- `--batch_size`: Batch size (default: 8)
- `--learning_rate`: Initial learning rate (default: 1e-4)
- `--max_epochs`: Maximum training epochs (default: 100)
- `--patience`: Early stopping patience (default: 100)
- `--num_workers`: DataLoader workers (default: 4)
- `--accumulate_grad_batches`: Gradient accumulation steps (default: 1)

**Loss Weights:**

- `--time_loss_weight`: Weight for time-domain L1 loss (default: 1.0)
- `--spectral_loss_weight`: Weight for spectral loss (default: 1.0)

**Checkpoints & Logging:**

- `--checkpoint_dir`: Directory for model checkpoints (default: `./checkpoints`)
- `--log_dir`: Directory for TensorBoard logs (default: `./logs`)
- `--resume_from`: Path to checkpoint for resuming training

**Precision:**

- `--precision`: Training precision - "32", "16", or "bf16" (default: "32")

### Monitoring Training

View training progress in TensorBoard:

```bash
tensorboard --logdir ./logs
```

Then open `http://localhost:6006/` in your browser.

The dashboard shows:

- Training/validation loss curves
- Audio samples (input, target, output) as waveforms

## Architecture Details

### Model Configuration

Default configuration in `train.py`:

```python
model = LiveifyModel(
    input_sr=int(args.sample_rate * args.segment_duration),  # 88,200 @ 4s
    output_sr=int(args.sample_rate * args.segment_duration),
    hidden_channels=128,
    encoder_strides=[8, 4, 4],
    transformer_dim=384,  # (512 + 256) // 2
    num_heads=8,
    num_layers=4,
    lr=args.learning_rate,
)
```

### Troubleshooting

### Whisper Dimension Mismatch Error

```
Warning: Whisper transcription failed for Starboy.mp3 (dimension mismatch)
```

**Solutions:**

- Check audio file integrity with `ffprobe`
- Re-encode corrupted files: `ffmpeg -i input.mp3 -acodec libmp3lame output.mp3`
- Skip problematic files and remove from dataset directories

### Out-of-Range Audio Warnings

The pipeline automatically normalizes audio to [-1, 1] at multiple stages:

- Dataset: Per-segment normalization
- Training: Safe logging normalization with clamping

If warnings persist, check audio file format compatibility.

### CUDA Out of Memory

Reduce batch size or segment duration:

```bash
python src/train.py --batch_size 4 --segment_duration 3.0
```

### Slow Dataset Loading

First run processes and caches segments. Subsequent runs load from cache (~100x faster).
Delete `./cache/` to force reprocessing if dataset changes.

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Submit a pull request with clear descriptions

## Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper) for automatic speech recognition
- [PyTorch Lightning](https://www.pytorchlightning.ai/) for training utilities
- [librosa](https://librosa.org/) for audio processing
