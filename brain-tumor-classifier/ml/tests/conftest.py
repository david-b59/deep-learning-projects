"""Shared test fixtures: a tiny fake MRI dataset built in a temporary folder."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from brain_mri.config import DataConfig

CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
N_TRAIN_PER_CLASS = 20
N_TEST_PER_CLASS = 6


def _write_image(path: Path, seed: int) -> None:
    """Write a small random grayscale JPEG (different seed, different content)."""
    rng = np.random.default_rng(seed)
    Image.fromarray(rng.integers(0, 256, (40, 50), dtype=np.uint8)).save(path)


@pytest.fixture
def raw_dir(tmp_path: Path) -> Path:
    """Fake dataset with Training/ and Testing/, one folder per class.

    Planted duplicates (used by the tests):
      - Testing/glioma/leak.jpg      is a copy of Training/glioma/img_0.jpg
      - Training/notumor/copy.jpg    is a copy of Training/notumor/img_0.jpg
    """
    root = tmp_path / "raw"
    seed = 0
    for split, count in (("Training", N_TRAIN_PER_CLASS), ("Testing", N_TEST_PER_CLASS)):
        for name in CLASSES:
            folder = root / split / name
            folder.mkdir(parents=True)
            for i in range(count):
                seed += 1
                _write_image(folder / f"img_{i}.jpg", seed)
    shutil.copy(root / "Training/glioma/img_0.jpg", root / "Testing/glioma/leak.jpg")
    shutil.copy(root / "Training/notumor/img_0.jpg", root / "Training/notumor/copy.jpg")
    return root


@pytest.fixture
def config(raw_dir: Path, tmp_path: Path) -> DataConfig:
    """Data configuration pointing to the fake dataset."""
    return DataConfig(
        raw_dir=raw_dir,
        processed_dir=tmp_path / "processed",
        image_size=32,
        batch_size=8,
        val_fraction=0.25,
        seed=0,
    )
