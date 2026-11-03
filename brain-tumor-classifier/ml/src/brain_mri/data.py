"""Data pipeline: transforms, datasets, stratified split and DataLoaders.

This is phase A of the course pipeline (``ImageFolder`` + ``transforms`` +
``DataLoader``), written as reusable, configurable and testable functions.

Splits:
    - ``train``: most of the ``Training`` folder (stratified), light augmentation.
    - ``val``: a stratified share of it (``val_fraction``), no augmentation. Used to
      tune and stop training.
    - ``test``: the ``Testing`` folder, no augmentation. Used once, at the end.

Duplicate images found by :mod:`brain_mri.duplicates` are excluded first, so the
same image cannot be on both sides of a split.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets
from torchvision.transforms import v2

from brain_mri.config import DataConfig
from brain_mri.duplicates import (
    IMAGE_SUFFIXES,
    TEST_DIR_NAME,
    TRAIN_DIR_NAME,
    load_or_create_report,
)

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class DataBundle:
    """Everything the training and evaluation code needs from the data."""

    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    class_names: list[str]
    sizes: dict[str, int]


def build_transforms(config: DataConfig, train: bool) -> v2.Compose:
    """Build the preprocessing (and, for training, augmentation) pipeline.

    Args:
        config: Data configuration.
        train: If True, add light random augmentation.

    Returns:
        A transform turning a PIL image into a normalized float tensor
        of shape ``(3, image_size, image_size)``.
    """
    size = (config.image_size, config.image_size)
    steps: list[v2.Transform] = [v2.ToImage(), v2.Resize(size, antialias=True)]
    if train:
        steps += [
            v2.RandomHorizontalFlip(p=config.flip_prob),
            v2.RandomRotation(degrees=config.rotation_degrees),
            v2.ColorJitter(brightness=config.jitter, contrast=config.jitter),
        ]
    steps += [
        v2.ToDtype(torch.float32, scale=True),  # uint8 [0, 255] -> float [0, 1]
        v2.Normalize(mean=config.mean, std=config.std),  # ImageNet statistics
    ]
    return v2.Compose(steps)


def make_image_folder(
    root: Path, raw_dir: Path, excluded: set[str], transform: v2.Compose
) -> datasets.ImageFolder:
    """Create an ``ImageFolder`` that skips the excluded (duplicate) files.

    Args:
        root: ``Training`` or ``Testing`` folder (one sub-folder per class).
        raw_dir: Folder the excluded paths are relative to.
        excluded: Relative POSIX paths to ignore.
        transform: Transform applied to each image.
    """

    def is_valid_file(path: str) -> bool:
        """Keep image files that are not in the exclusion list."""
        file = Path(path)
        relative = file.relative_to(raw_dir).as_posix()
        return file.suffix.lower() in IMAGE_SUFFIXES and relative not in excluded

    return datasets.ImageFolder(root, transform=transform, is_valid_file=is_valid_file)


def stratified_split(
    targets: list[int], val_fraction: float, seed: int
) -> tuple[list[int], list[int]]:
    """Split sample indices into train and validation, keeping class proportions.

    Args:
        targets: Class index of each sample.
        val_fraction: Share of samples for validation, in (0, 1).
        seed: Random seed (same seed, same split).

    Returns:
        ``(train_indices, val_indices)``, disjoint and sorted.
    """
    train_idx, val_idx = train_test_split(
        np.arange(len(targets)), test_size=val_fraction, stratify=targets, random_state=seed
    )
    return sorted(train_idx.tolist()), sorted(val_idx.tolist())


def make_datasets(config: DataConfig) -> tuple[Dataset, Dataset, Dataset, list[str]]:
    """Build the train, validation and test datasets.

    The training folder is loaded twice (with and without augmentation) and the
    same stratified indices are applied to both, so validation is never augmented.

    Returns:
        ``(train, val, test, class_names)`` with sorted class names.
    """
    report = load_or_create_report(config.raw_dir, config.processed_dir)
    excluded = report.files_to_exclude()
    LOGGER.info("Excluding %d duplicate file(s)", len(excluded))

    train_root = config.raw_dir / TRAIN_DIR_NAME
    test_root = config.raw_dir / TEST_DIR_NAME
    train_full = make_image_folder(
        train_root, config.raw_dir, excluded, build_transforms(config, train=True)
    )
    val_full = make_image_folder(
        train_root, config.raw_dir, excluded, build_transforms(config, train=False)
    )
    test = make_image_folder(
        test_root, config.raw_dir, excluded, build_transforms(config, train=False)
    )
    if train_full.classes != test.classes:
        raise ValueError(f"Class mismatch: {train_full.classes} vs {test.classes}")

    train_idx, val_idx = stratified_split(train_full.targets, config.val_fraction, config.seed)
    return Subset(train_full, train_idx), Subset(val_full, val_idx), test, train_full.classes


def _seed_worker(worker_id: int) -> None:
    """Seed Python and NumPy in each DataLoader worker.

    The ``v2`` augmentations draw from torch's random generator, which PyTorch seeds
    per worker itself. For fully reproducible augmentation, the training script must
    also call ``torch.manual_seed(config.seed)`` (done in step 3).
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_dataloaders(config: DataConfig) -> DataBundle:
    """Build the three DataLoaders. Only the training one is shuffled.

    Args:
        config: Data configuration.

    Returns:
        A :class:`DataBundle` with the loaders, class names and split sizes.
    """
    train, val, test, class_names = make_datasets(config)
    generator = torch.Generator().manual_seed(config.seed)  # reproducible shuffling

    def loader(dataset: Dataset, shuffle: bool) -> DataLoader:
        """Create a DataLoader with the shared settings."""
        return DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=shuffle,
            num_workers=config.num_workers,
            worker_init_fn=_seed_worker,
            generator=generator if shuffle else None,
        )

    sizes = {"train": len(train), "val": len(val), "test": len(test)}
    LOGGER.info("Split sizes: %s | classes: %s", sizes, class_names)
    return DataBundle(
        train_loader=loader(train, shuffle=True),
        val_loader=loader(val, shuffle=False),
        test_loader=loader(test, shuffle=False),
        class_names=class_names,
        sizes=sizes,
    )
