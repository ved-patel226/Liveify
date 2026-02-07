from dataset.dataset import StudioLiveDataModule


if __name__ == "__main__":
    datamodule = StudioLiveDataModule(
        studio_dir="./dataset/studio",
        live_dir="./dataset/live",
        development_mode=True,
        segment_duration=5.0,
    )
    datamodule.setup()
    train_loader = datamodule.train_dataloader()
    for i, item in enumerate(next(iter(train_loader))):
        print(f"Batch item {i} shape: {item.shape}")
