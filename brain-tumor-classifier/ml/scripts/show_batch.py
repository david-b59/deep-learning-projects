"""Inspect the data pipeline: split sizes, class balance and one training batch.

Prints the tensor shapes and labels of the first training batch and saves an
image grid (``data/processed/batch.png``, git-ignored) so the augmentation and
the labels can be checked by eye.

Usage (from ``brain-tumor-classifier/ml/``)::

    uv run python scripts/show_batch.py
"""

from __future__ import annotations

import logging
from collections import Counter

import matplotlib

matplotlib.use("Agg")  # no window: write the figure to a file
import matplotlib.pyplot as plt
import torch

from brain_mri.config import load_config
from brain_mri.data import build_dataloaders

N_SHOWN = 8


def class_counts(loader: torch.utils.data.DataLoader, class_names: list[str]) -> dict[str, int]:
    """Count the samples of each class in a loader's dataset (no image loading)."""
    dataset = loader.dataset
    base = dataset.dataset if hasattr(dataset, "dataset") else dataset
    indices = dataset.indices if hasattr(dataset, "indices") else range(len(base))
    counts = Counter(base.targets[i] for i in indices)
    return {class_names[k]: counts[k] for k in sorted(counts)}


def main() -> None:
    """Build the DataLoaders and display one training batch."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config = load_config().data
    bundle = build_dataloaders(config)

    print("Classes:", bundle.class_names)
    for name, loader in (
        ("train", bundle.train_loader),
        ("val", bundle.val_loader),
        ("test", bundle.test_loader),
    ):
        print(
            f"{name:5s} {bundle.sizes[name]:5d} images | {class_counts(loader, bundle.class_names)}"
        )

    images, labels = next(iter(bundle.train_loader))
    print("\nBatch images:", tuple(images.shape), images.dtype)
    print("Batch labels:", tuple(labels.shape), labels.dtype)
    print("Labels      :", labels[:N_SHOWN].tolist())
    print(f"Pixel range : min={images.min():.2f} max={images.max():.2f} mean={images.mean():.2f}")

    # Undo the normalization to display the images with natural colors.
    mean = torch.tensor(config.mean).view(1, 3, 1, 1)
    std = torch.tensor(config.std).view(1, 3, 1, 1)
    shown = (images[:N_SHOWN] * std + mean).clamp(0, 1)

    fig, axes = plt.subplots(2, 4, figsize=(10, 5.5))
    for ax, image, label in zip(axes.ravel(), shown, labels[:N_SHOWN], strict=True):
        ax.imshow(image.permute(1, 2, 0))  # (C, H, W) -> (H, W, C)
        ax.set_title(bundle.class_names[label], fontsize=9)
        ax.axis("off")
    plt.tight_layout()
    config.processed_dir.mkdir(parents=True, exist_ok=True)
    output = config.processed_dir / "batch.png"
    fig.savefig(output, dpi=100)
    print(f"Figure saved to {output}")


if __name__ == "__main__":
    main()
