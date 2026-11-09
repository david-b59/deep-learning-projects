"""Prétraitement des images : redimensionnement, augmentation légère et normalisation.

Module volontairement léger (torch et torchvision seulement) : il est partagé par
l'entraînement (:mod:`brain_mri.data`) et par la prédiction (:mod:`brain_mri.predict`).

Deux modes (``DataConfig.preprocessing``) :
    - ``legacy`` : l'image est étirée directement en carré (comportement historique, celui du
      modèle en ligne). Ne pas le modifier : il sert à reproduire ce modèle (variante A0).
    - ``head_crop`` : tête recadrée par :func:`brain_mri.head_crop.crop_head` (déjà carrée,
      avec des bandes noires, donc sans étirement), puis 224x224, puis normalisation ImageNet.
      L'augmentation d'entraînement (types vérifiés dans torchvision 0.29.2) travaille sur des
      images 8 bits : géométrie, contraste, gamma, flou, JPEG, puis conversion en décimaux et bruit.
"""

from __future__ import annotations

import torch
from PIL import Image
from torch import nn
from torchvision.transforms import v2

from brain_mri.config import DataConfig, HeadCropAugConfig
from brain_mri.head_crop import crop_head


def _crop_head_step(image: Image.Image) -> Image.Image:
    """Étape ``head_crop`` : recadre la tête en carré (niveaux de gris, bandes noires)."""
    return crop_head(image)


def _to_rgb_step(image: Image.Image) -> Image.Image:
    """Étape ``head_crop`` : recopie le niveau de gris sur 3 canaux (le ResNet en attend 3)."""
    return image.convert("RGB")


class RandomGamma(nn.Module):
    """Gamma aléatoire sur une image 8 bits (torchvision v2 n'a pas de transformation prête).

    Un gamma < 1 éclaircit les tons moyens, un gamma > 1 les assombrit. Le tirage utilise
    le générateur de torch : ``torch.manual_seed`` rend donc l'augmentation reproductible.
    """

    def __init__(self, gamma_range: tuple[float, float]) -> None:
        """Mémorise la plage de gamma.

        Args:
            gamma_range: Bornes (min, max) du gamma tiré uniformément.
        """
        super().__init__()
        self.gamma_range = gamma_range

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Applique un gamma tiré au hasard à un tenseur d'image.

        Args:
            image: Tenseur d'image ``(C, H, W)`` (uint8 ou décimal 0-1).

        Returns:
            L'image avec le gamma appliqué (même type).
        """
        low, high = self.gamma_range
        gamma = low + (high - low) * torch.rand(1).item()
        return v2.functional.adjust_gamma(image, gamma)


def _head_crop_augmentation(aug: HeadCropAugConfig, size: tuple[int, int]) -> list:
    """Liste des étapes d'augmentation du mode ``head_crop`` jusqu'au JPEG inclus.

    Ordre : dézoom (bandes noires) -> recadrage aléatoire redimensionné à ``size`` -> miroir ->
    rotation -> contraste -> gamma -> flou -> JPEG. Tout reste en uint8 (le JPEG l'exige).

    Args:
        aug: Paramètres d'augmentation.
        size: Taille de sortie ``(hauteur, largeur)``.

    Returns:
        Les transformations, à placer après ``ToImage`` et avant ``ToDtype``.
    """
    return [
        v2.RandomZoomOut(fill=0, side_range=(1.0, aug.zoomout_max), p=aug.zoomout_prob),
        v2.RandomResizedCrop(size, scale=aug.crop_scale, ratio=aug.crop_ratio, antialias=True),
        v2.RandomHorizontalFlip(p=aug.flip_prob),
        v2.RandomRotation(degrees=aug.rotation_degrees),  # fond noir (fill=0 par défaut)
        v2.ColorJitter(contrast=aug.contrast),
        RandomGamma(aug.gamma_range),
        v2.RandomApply([v2.GaussianBlur(kernel_size=5, sigma=aug.blur_sigma)], p=aug.blur_prob),
        v2.RandomApply([v2.JPEG(quality=aug.jpeg_quality)], p=aug.jpeg_prob),
    ]


def _build_head_crop_transforms(config: DataConfig, train: bool) -> v2.Compose:
    """Construit la chaîne du mode ``head_crop`` (voir :func:`build_transforms`)."""
    size = (config.image_size, config.image_size)
    steps: list = [v2.Lambda(_crop_head_step), v2.Lambda(_to_rgb_step), v2.ToImage()]
    if train:
        steps += _head_crop_augmentation(config.head_crop_aug, size)
    else:
        steps.append(v2.Resize(size, antialias=True))  # l'image est déjà carrée : pas d'étirement
    steps.append(v2.ToDtype(torch.float32, scale=True))  # entiers 0-255 -> décimaux 0-1
    if train:
        # Le bruit se règle sur l'échelle 0-1 : il vient donc après la conversion en décimaux.
        noise = v2.GaussianNoise(sigma=config.head_crop_aug.noise_sigma)
        steps.append(v2.RandomApply([noise], p=config.head_crop_aug.noise_prob))
    steps.append(v2.Normalize(mean=config.mean, std=config.std))  # statistiques d'ImageNet
    return v2.Compose(steps)


def build_transforms(config: DataConfig, train: bool) -> v2.Compose:
    """Construit la chaîne de prétraitement (et d'augmentation, pour l'entraînement).

    Args:
        config: Configuration des données.
        train: Si True, ajoute une augmentation aléatoire (légère en mode ``legacy``, voir
            ``head_crop_aug`` en mode ``head_crop``).

    Returns:
        Une transformation qui convertit une image PIL en tenseur décimal normalisé
        de forme ``(3, image_size, image_size)``.
    """
    if config.preprocessing == "head_crop":
        return _build_head_crop_transforms(config, train)
    # Mode "legacy" : code historique, volontairement inchangé.
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
