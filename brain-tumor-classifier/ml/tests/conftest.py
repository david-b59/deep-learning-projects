"""Fixtures de test partagées : un minuscule faux jeu d'IRM créé dans un dossier temporaire."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from brain_mri.config import DataConfig

CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
N_TRAIN_PER_CLASS = 20  # images d'entraînement par classe dans le faux jeu
N_TEST_PER_CLASS = 6  # images de test par classe dans le faux jeu


def _write_image(path: Path, seed: int) -> None:
    """Écrit un petit JPEG aléatoire en niveaux de gris (graine différente = contenu différent)."""
    rng = np.random.default_rng(seed)
    Image.fromarray(rng.integers(0, 256, (40, 50), dtype=np.uint8)).save(path)


@pytest.fixture
def raw_dir(tmp_path: Path) -> Path:
    """Crée un faux jeu de données avec Training/ et Testing/, un dossier par classe.

    Doublons plantés (utilisés par les tests) :
      - Testing/glioma/leak.jpg      est une copie de Training/glioma/img_0.jpg
      - Training/notumor/copy.jpg    est une copie de Training/notumor/img_0.jpg

    Returns:
        Le dossier racine du faux jeu de données.
    """
    root = tmp_path / "raw"
    seed = 0
    for split, count in (("Training", N_TRAIN_PER_CLASS), ("Testing", N_TEST_PER_CLASS)):
        for name in CLASSES:
            folder = root / split / name
            folder.mkdir(parents=True)
            for i in range(count):
                seed += 1  # une graine différente par image : tous les contenus diffèrent
                _write_image(folder / f"img_{i}.jpg", seed)
    shutil.copy(root / "Training/glioma/img_0.jpg", root / "Testing/glioma/leak.jpg")
    shutil.copy(root / "Training/notumor/img_0.jpg", root / "Training/notumor/copy.jpg")
    return root


@pytest.fixture
def config(raw_dir: Path, tmp_path: Path) -> DataConfig:
    """Configuration des données qui pointe vers le faux jeu (images 32x32, lots de 8)."""
    return DataConfig(
        raw_dir=raw_dir,
        processed_dir=tmp_path / "processed",
        image_size=32,
        batch_size=8,
        val_fraction=0.25,
        seed=0,
    )
