"""
Main module for PC Classifier
"""

import re
from pathlib import Path
import torch
from torch.utils.data import DataLoader, RandomSampler
from torch.amp import autocast, GradScaler
from collections import deque

from pc_classifier.dataset import PointsDataset
from pc_classifier.neuralnetcode import PointNet
from pc_classifier.transforms import (
    Drop,
    RandomRotate,
    RandomMirror,
    Jitter,
    Transforms,
)


base_epoch = 0


def main() -> None:
    """
    Main function
    """
    transforms: list[Transforms] = [
        Drop(0.2),
        RandomRotate(),
        RandomMirror(),
        Jitter(0.01),
    ]

    # Pass the root directories that contain all sequence subfolders
    points_dataset = PointsDataset(
        "training_data/data_odometry_velodyne/dataset/sequences",
        "training_data/data_odometry_labels/dataset/sequences",
        num_classes=25,
        transforms=transforms,
    )

    train(points_dataset, num_classes=25, epochs=5000, batch_size=512)


def load_model(num_classes: int) -> PointNet:
    """Load the model

    Args:
        num_classes (int): Number of classes

    Returns:
        PointNet: The model
    """

    model = PointNet(num_classes)
    base_epoch = 0

    models_dir = Path("models")
    latest_path = None
    latest_num = -1

    if models_dir.exists():
        for p in models_dir.glob("model_*.pth"):
            m = re.match(r"model_(\d+)\.pth$", p.name)
            if not m:
                continue
            n = int(m.group(1))
            if n > latest_num:
                latest_num = n
                latest_path = p

    if latest_path is not None:
        model.load_state_dict(torch.load(str(latest_path), weights_only=True))
        print(f"[train] Loaded weights from {latest_path}")

        # This is used to continue training from the last checkpoint
        base_epoch = int(latest_path.stem.split("_")[1]) + 1

    return model, base_epoch


def train(
    points_dataset: PointsDataset,
    num_classes: int,
    epochs: int = 2,
    batch_size: int = 1024,
    early_stopping_patience: int | None = 20,
    early_stopping_min_delta: float = 1e-4,
):
    """Train the model

    Args:
        points_dataset (PointsDataset): The points dataset
        num_classes (int): The number of classes
        epochs (int, optional): The number of epochs. Defaults to 2.
        batch_size (int, optional): The batch size. Defaults to 1024.
        transforms (list[Transforms], optional): The transforms. Defaults to [].
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    print(f"[train] Using device: {device}")

    print("[train] Initializing model...")
    model, base_epoch = load_model(num_classes)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    sampler = RandomSampler(points_dataset, replacement=True, num_samples=200_000)
    scaler = GradScaler()

    loader = DataLoader(
        points_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=12,
        prefetch_factor=2,
    )
    print(
        "[train] Model ready. Dataset size:",
        len(points_dataset),
        "Batches:",
        len(loader),
    )

    # Ensure models directory exists for checkpoints
    Path("models").mkdir(parents=True, exist_ok=True)

    # Early stopping trackers
    best_epoch_average_loss = float("inf")
    best_epoch_index = None
    epochs_without_improvement = 0

    moving_labels = deque(maxlen=100)
    moving_logits = deque(maxlen=100)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for step, batch in enumerate(loader, start=1):
            points, labels = batch  # points: (B, 1024, 3), labels: (B, num_classes)

            # Converting to torch tensors and moving to device
            points = points.to(device, dtype=torch.float32)  # (B, 1024, 3)
            labels = labels.to(device).argmax(dim=1).long()  # (B, num_classes)

            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=device.type, dtype=torch.bfloat16):
                optimizer.zero_grad()
                logits = model(points)  # (B, 1024, 3)
                loss = criterion(logits, labels)

            if not torch.isfinite(loss):
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            moving_labels.append(labels.to("cpu"))
            moving_logits.append(logits.to("cpu"))

            running_loss += loss.item()
            if step % 50 == 0 or step == len(loader):
                avg = running_loss / step
                # Compute average accuracy over recent batches
                total = 0
                correct = 0
                for label, logit in zip(moving_labels, moving_logits):
                    preds = logit.argmax(dim=1)
                    correct += (preds == label).sum().item()
                    total += label.numel()

                acc = correct / max(1, total)
                print(f"[train] Average correct classification rate: {acc:.4f}")
                print(
                    f"[train] Epoch {epoch}/{epochs} Step {step}/{len(loader)} Loss {avg:.4f}",
                )
        epoch_average_loss = running_loss / max(1, len(loader))
        torch.save(model.state_dict(), f"models/model_{base_epoch+epoch}.pth")
        print(
            f"[train] Epoch {base_epoch+epoch} completed. Avg loss: {epoch_average_loss:.4f}"
        )

        # Early stopping logic on training loss (no separate validation set available)
        if best_epoch_average_loss - epoch_average_loss > early_stopping_min_delta:
            best_epoch_average_loss = epoch_average_loss
            best_epoch_index = base_epoch + epoch
            epochs_without_improvement = 0
            # Save/update best model snapshot
            torch.save(model.state_dict(), "models/best_model.pth")
            print(
                f"[train] New best model at epoch {best_epoch_index} with loss {best_epoch_average_loss:.4f}"
            )
        else:
            epochs_without_improvement += 1
            if (
                early_stopping_patience is not None
                and epochs_without_improvement >= early_stopping_patience
            ):
                print(
                    f"[train] Early stopping triggered after {epochs_without_improvement} epochs without improvement. Best epoch: {best_epoch_index} loss: {best_epoch_average_loss:.4f}"
                )
                break


if __name__ == "__main__":
    main()
