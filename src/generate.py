"""
Inference and generation utilities for the Liveify model.

Supports:
- Single audio file generation
- Batch generation over directories
- Streaming generation for real-time applications
"""

import torch
import librosa
import soundfile as sf
import os
from pathlib import Path
from typing import Optional, List
import argparse

from models import EncodecLatentModel, EncodecLatentLightningModule


def generate_from_checkpoint(
    checkpoint_path: str,
    studio_audio: torch.Tensor,
    n_steps: int = 48,
    temperature: float = 0.8,
    context_length: int = 12,
    forward_context_length: int = 24,
    decode_strategy: str = "sample",
    device: str = "cuda",
) -> dict:
    """Generate live audio from studio audio using a trained model.

    Args:
        checkpoint_path: Path to the trained checkpoint
        studio_audio: Studio audio tensor (B, L) or (L,)
        n_steps: Number of steps to generate
        temperature: Sampling temperature (0.5-2.0)
        context_length: Context length used during training
        forward_context_length: Forward context length used during training
        decode_strategy: "sample", "argmax", or "deterministic"
        device: "cuda" or "cpu"

    Returns:
        Dictionary with generated latents and audio
    """
    # Ensure audio is batched
    if studio_audio.dim() == 1:
        studio_audio = studio_audio.unsqueeze(0)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Reconstruct model
    model = EncodecLatentModel(
        latent_dim=128,
        context_length=context_length,
        forward_context_length=forward_context_length,
        d_model=128,
        num_heads=4,
        intra_layers=2,
        inter_layers=4,
        final_cross_layers=2,
        ff_mult=2,
        dropout=0.0,  # No dropout in inference
        drop_path=0.0,
    )

    lightning_module = EncodecLatentLightningModule(
        model=model,
        sample_rate=48000,
        encodec_sample_rate=24000,
    )
    lightning_module.load_state_dict(checkpoint["state_dict"])
    lightning_module.to(device)
    lightning_module.eval()

    # Generate
    result = lightning_module.generate(
        studio_audio=studio_audio.to(device),
        n_steps=n_steps,
        temperature=temperature,
        decode_strategy=decode_strategy,
        return_audio=True,
    )

    return result


def generate_single_file(
    checkpoint_path: str,
    studio_audio_path: str,
    output_path: str = "generated.wav",
    n_steps: int = 48,
    temperature: float = 0.8,
    context_length: int = 12,
    forward_context_length: int = 24,
    device: str = "cuda",
):
    """Generate audio for a single studio audio file.

    Args:
        checkpoint_path: Path to trained checkpoint
        studio_audio_path: Path to input studio audio
        output_path: Path to save generated audio
        n_steps: Number of steps to generate
        temperature: Sampling temperature
        context_length: Context length from training
        forward_context_length: Forward context from training
        device: "cuda" or "cpu"
    """
    # Load studio audio
    studio_audio, sr = librosa.load(studio_audio_path, sr=24000, mono=True)
    studio_tensor = torch.from_numpy(studio_audio).float()

    print(f"Loaded studio audio: {studio_audio_path}")
    print(f"  Shape: {studio_tensor.shape}")
    print(f"  Duration: {len(studio_tensor) / 24000:.2f}s")

    # Generate
    print(f"\nGenerating {n_steps} steps with temperature={temperature}...")
    result = generate_from_checkpoint(
        checkpoint_path,
        studio_tensor,
        n_steps=n_steps,
        temperature=temperature,
        context_length=context_length,
        forward_context_length=forward_context_length,
        device=device,
    )

    # Save output
    generated_audio = result["generated_audio"][0, 0].cpu().numpy()
    sf.write(output_path, generated_audio, 24000)

    print(f"\nGenerated audio saved: {output_path}")
    print(f"  Duration: {len(generated_audio) / 24000:.2f}s")

    return result


def batch_generate(
    checkpoint_path: str,
    studio_dir: str,
    output_dir: str,
    n_steps: int = 48,
    temperature: float = 0.8,
    context_length: int = 12,
    device: str = "cuda",
):
    """Generate for all audio files in a directory.

    Args:
        checkpoint_path: Path to trained checkpoint
        studio_dir: Directory containing studio audio files
        output_dir: Directory to save generated audio
        n_steps: Number of steps to generate
        temperature: Sampling temperature
        context_length: Context length from training
        device: "cuda" or "cpu"
    """
    os.makedirs(output_dir, exist_ok=True)

    files = sorted([f for f in os.listdir(studio_dir) if f.endswith(".wav")])
    print(f"Found {len(files)} .wav files in {studio_dir}")

    for idx, filename in enumerate(files, 1):
        studio_path = os.path.join(studio_dir, filename)
        output_path = os.path.join(output_dir, filename)

        print(f"\n[{idx}/{len(files)}] Processing: {filename}")

        try:
            generate_single_file(
                checkpoint_path,
                studio_path,
                output_path,
                n_steps=n_steps,
                temperature=temperature,
                context_length=context_length,
                device=device,
            )
        except Exception as e:
            print(f"  ERROR: {e}")
            continue


def streaming_generation_example(
    checkpoint_path: str,
    studio_audio: torch.Tensor,
    context_length: int = 12,
    device: str = "cuda",
):
    """Example of streaming generation for real-time applications.

    This generates one latent step at a time, using previous predictions
    as context for the next step.

    Args:
        checkpoint_path: Path to trained checkpoint
        studio_audio: Full studio audio (L,)
        context_length: Context length from training
        device: "cuda" or "cpu"
    """
    # Load model
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = EncodecLatentModel(
        latent_dim=128,
        context_length=context_length,
        forward_context_length=24,
        d_model=128,
        num_heads=4,
        intra_layers=2,
        inter_layers=4,
        final_cross_layers=2,
        ff_mult=2,
    )
    lightning_module = EncodecLatentLightningModule(model=model)
    lightning_module.load_state_dict(checkpoint["state_dict"])
    lightning_module.to(device)

    # Ensure audio is batched
    if studio_audio.dim() == 1:
        studio_audio = studio_audio.unsqueeze(0)

    # Encode studio once
    x_studio = lightning_module._encode_audio(studio_audio.to(device))[0]
    B, S, C, T = x_studio.shape

    print(f"Studio latents shape: {x_studio.shape}")

    # Streaming generation
    generated_cache = []
    all_generated = []

    print("\nStarting streaming generation...")
    for step in range(S):  # Generate as many steps as studio has
        studio_chunk = x_studio[:, step]

        # Generate next latent
        next_latent = lightning_module.generate_streaming(
            studio_chunk, generated_cache, temperature=0.8
        )

        generated_cache.append(next_latent)
        all_generated.append(next_latent)

        if (step + 1) % 10 == 0:
            print(f"  Generated {step + 1} steps...")

    generated_audio = torch.stack(all_generated, dim=1)
    print(f"Final generated shape: {generated_audio.shape}")

    return generated_audio


def main():
    parser = argparse.ArgumentParser(
        description="Generate audio using trained Liveify model"
    )

    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to trained checkpoint"
    )
    parser.add_argument(
        "--input", type=str, required=True, help="Input studio audio file or directory"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="generated.wav",
        help="Output path (file or directory)",
    )
    parser.add_argument(
        "--n-steps", type=int, default=48, help="Number of steps to generate"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.8, help="Sampling temperature"
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=12,
        help="Context length (must match training)",
    )
    parser.add_argument(
        "--forward-context",
        type=int,
        default=24,
        help="Forward context length (must match training)",
    )
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument(
        "--strategy",
        type=str,
        default="sample",
        choices=["sample", "argmax", "deterministic"],
    )

    args = parser.parse_args()

    # Check if input is file or directory
    if os.path.isdir(args.input):
        print(f"Batch mode: {args.input} -> {args.output}")
        batch_generate(
            args.checkpoint,
            args.input,
            args.output,
            n_steps=args.n_steps,
            temperature=args.temperature,
            context_length=args.context_length,
            device=args.device,
        )
    else:
        print(f"Single file mode: {args.input} -> {args.output}")
        generate_single_file(
            args.checkpoint,
            args.input,
            args.output,
            n_steps=args.n_steps,
            temperature=args.temperature,
            context_length=args.context_length,
            forward_context_length=args.forward_context,
            device=args.device,
        )


if __name__ == "__main__":
    main()
