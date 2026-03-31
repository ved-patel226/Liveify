import wandb
import torch
import gc
import sys
from train import train, parse_args


def sweep():
    run = wandb.init()

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    old_argv = sys.argv
    sys.argv = [old_argv[0]]
    args = parse_args()
    sys.argv = old_argv

    args.max_epochs = 100
    args.logger = "wandb"

    for key, value in wandb.config.items():
        if hasattr(args, key):
            setattr(args, key, value)

    try:
        train(args)
    except KeyboardInterrupt:
        print("Keyboard interrupt detected. Exiting the run.")
        raise
    finally:
        run.finish()

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    sweep_config = {
        "method": "bayes",
        "metric": {"name": "val/loss", "goal": "minimize"},
        "parameters": {
            "learning_rate": {
                "distribution": "log_uniform_values",
                "min": 1e-6,
                "max": 1e-3,
            },
            "batch_size": {"values": [4, 8]},
            "d_model": {"values": [128, 256, 384]},
            "num_heads": {"values": [2, 4, 8]},
            "latent_layers": {"values": [2, 3, 4]},
            "dropout": {"values": [0.1, 0.15, 0.2, 0.3]},
            "ff_mult": {"values": [1, 2, 4]},
            "segment_overlap": {"values": [0.0, 0.25, 0.4, 0.5, 0.75, 0.9]},
            "context_mask_prob": {"values": [0.0, 0.1, 0.2]},
            "decode_loss_freq": {"values": [2, 4, 8]},
        },
    }

    sweep_id = wandb.sweep(sweep_config, project="liveify-sweep")

    print(f"Started sweep: {sweep_id}")
    try:
        wandb.agent(sweep_id, sweep, count=None)
    except KeyboardInterrupt:
        print("\nSweep interrupted by user. Exiting.")
