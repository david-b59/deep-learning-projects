"""Inspecte le pipeline de données : tailles des jeux, équilibre des classes et un lot.

Affiche les formes et les étiquettes du premier lot d'entraînement, et enregistre une
grille d'images (``data/processed/batch.png``, hors git) pour contrôler à l'œil
l'augmentation et les étiquettes.

Utilisation (depuis ``brain-tumor-classifier/ml/``) ::

    uv run python scripts/show_batch.py
"""

from __future__ import annotations

import logging
from collections import Counter

import matplotlib

matplotlib.use("Agg")  # pas de fenêtre : la figure est écrite dans un fichier
import matplotlib.pyplot as plt
import torch

from brain_mri.config import load_config
from brain_mri.data import build_dataloaders

N_SHOWN = 8  # nombre d'images affichées dans la grille


def class_counts(loader: torch.utils.data.DataLoader, class_names: list[str]) -> dict[str, int]:
    """Compte les échantillons de chaque classe d'un DataLoader (sans charger les images)."""
    dataset = loader.dataset
    # Les jeux d'entraînement et de validation sont des Subset : on remonte au jeu de base.
    base = dataset.dataset if hasattr(dataset, "dataset") else dataset
    indices = dataset.indices if hasattr(dataset, "indices") else range(len(base))
    counts = Counter(base.targets[i] for i in indices)
    return {class_names[k]: counts[k] for k in sorted(counts)}


def main() -> None:
    """Construit les DataLoaders et affiche un lot d'entraînement."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config = load_config().data
    bundle = build_dataloaders(config)

    print("Classes :", bundle.class_names)
    for name, loader in (
        ("train", bundle.train_loader),
        ("val", bundle.val_loader),
        ("test", bundle.test_loader),
    ):
        print(
            f"{name:5s} {bundle.sizes[name]:5d} images | {class_counts(loader, bundle.class_names)}"
        )

    images, labels = next(iter(bundle.train_loader))
    print("\nImages du lot    :", tuple(images.shape), images.dtype)
    print("Étiquettes du lot:", tuple(labels.shape), labels.dtype)
    print("Étiquettes       :", labels[:N_SHOWN].tolist())
    print(
        f"Valeurs de pixels: min={images.min():.2f} max={images.max():.2f} moy={images.mean():.2f}"
    )

    # On annule la normalisation pour afficher les images avec leurs couleurs naturelles.
    mean = torch.tensor(config.mean).view(1, 3, 1, 1)
    std = torch.tensor(config.std).view(1, 3, 1, 1)
    shown = (images[:N_SHOWN] * std + mean).clamp(0, 1)

    fig, axes = plt.subplots(2, 4, figsize=(10, 5.5))
    for ax, image, label in zip(axes.ravel(), shown, labels[:N_SHOWN], strict=True):
        ax.imshow(image.permute(1, 2, 0))  # (C, H, W) -> (H, W, C) pour matplotlib
        ax.set_title(bundle.class_names[label], fontsize=9)
        ax.axis("off")
    plt.tight_layout()
    config.processed_dir.mkdir(parents=True, exist_ok=True)
    output = config.processed_dir / "batch.png"
    fig.savefig(output, dpi=100)
    print(f"Figure enregistrée dans {output}")


if __name__ == "__main__":
    main()
