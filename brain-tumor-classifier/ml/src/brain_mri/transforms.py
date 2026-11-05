"""Prétraitement des images : redimensionnement, augmentation légère et normalisation.

Module volontairement léger (torch et torchvision seulement) : il est partagé par
l'entraînement (:mod:`brain_mri.data`) et par la prédiction (:mod:`brain_mri.predict`).
"""

from __future__ import annotations

import torch
from torchvision.transforms import v2

from brain_mri.config import DataConfig


def build_transforms(config: DataConfig, train: bool) -> v2.Compose:
    """Construit la chaîne de prétraitement (et d'augmentation, pour l'entraînement).

    Args:
        config: Configuration des données.
        train: Si True, ajoute une augmentation aléatoire légère.

    Returns:
        Une transformation qui convertit une image PIL en tenseur décimal normalisé
        de forme ``(3, image_size, image_size)``.
    """
    size = (config.image_size, config.image_size)
    steps: list[v2.Transform] = [v2.ToImage(), v2.Resize(size, antialias=True)]
    if train:
        # Augmentation : on montre au modèle des variantes légères de chaque image.
        steps += [
            v2.RandomHorizontalFlip(p=config.flip_prob),
            v2.RandomRotation(degrees=config.rotation_degrees),
            v2.ColorJitter(brightness=config.jitter, contrast=config.jitter),
        ]
    steps += [
        v2.ToDtype(torch.float32, scale=True),  # entiers 0-255 -> décimaux 0-1
        v2.Normalize(mean=config.mean, std=config.std),  # statistiques d'ImageNet
    ]
    return v2.Compose(steps)
