"""Tests de brain_mri.transforms : mode legacy inchangé, mode head_crop, augmentation."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image
from pydantic import ValidationError
from torchvision.transforms import v2

from brain_mri.config import DataConfig
from brain_mri.transforms import RandomGamma, build_transforms


def _legacy_reference(config: DataConfig, train: bool) -> v2.Compose:
    """Recopie à la main la chaîne d'origine (avant l'ajout des modes) pour comparer."""
    size = (config.image_size, config.image_size)
    steps: list = [v2.ToImage(), v2.Resize(size, antialias=True)]
    if train:
        steps += [
            v2.RandomHorizontalFlip(p=config.flip_prob),
            v2.RandomRotation(degrees=config.rotation_degrees),
            v2.ColorJitter(brightness=config.jitter, contrast=config.jitter),
        ]
    steps += [v2.ToDtype(torch.float32, scale=True), v2.Normalize(mean=config.mean, std=config.std)]
    return v2.Compose(steps)


def _ellipse_image(width: int = 200, height: int = 120, a: int = 80, b: int = 30) -> Image.Image:
    """Image RGB rectangulaire : une ellipse claire (demi-axes a, b) sur fond noir."""
    y, x = np.mgrid[0:height, 0:width]
    inside = ((x - width / 2) / a) ** 2 + ((y - height / 2) / b) ** 2 <= 1
    return Image.fromarray((inside * 200).astype(np.uint8)).convert("RGB")


def _noisy_image() -> Image.Image:
    """Image RGB 60x90 avec une ellipse texturée (pour tester l'augmentation)."""
    rng = np.random.default_rng(0)
    base = np.asarray(_ellipse_image(90, 60, 35, 20).convert("L"), dtype=np.int16)
    noisy = np.clip(base + rng.integers(0, 40, base.shape), 0, 255).astype(np.uint8)
    return Image.fromarray(noisy).convert("RGB")


def _denormalize(tensor: torch.Tensor, config: DataConfig) -> torch.Tensor:
    """Annule la normalisation ImageNet et renvoie le premier canal (valeurs 0-1)."""
    mean = torch.tensor(config.mean).view(3, 1, 1)
    std = torch.tensor(config.std).view(3, 1, 1)
    return (tensor * std + mean)[0]


def _bright_ratio(gray: torch.Tensor) -> float:
    """Rapport largeur / hauteur de la zone claire (> 0,5) d'une image 2D."""
    ys, xs = torch.where(gray > 0.5)
    return (xs.max() - xs.min() + 1).item() / (ys.max() - ys.min() + 1).item()


def test_default_mode_is_legacy() -> None:
    """Le prétraitement par défaut est « legacy » et le manifeste est absent."""
    config = DataConfig()
    assert config.preprocessing == "legacy"
    assert config.split_manifest is None


def test_unknown_preprocessing_rejected() -> None:
    """Un mode inconnu est refusé par la configuration."""
    with pytest.raises(ValidationError):
        DataConfig(preprocessing="autre")


def test_legacy_is_unchanged() -> None:
    """Le mode legacy donne le même tenseur que la chaîne d'origine, avec et sans augmentation."""
    config = DataConfig(image_size=32)
    image = _noisy_image()
    assert torch.equal(
        build_transforms(config, train=False)(image), _legacy_reference(config, False)(image)
    )
    torch.manual_seed(3)  # avec augmentation : même graine, même résultat
    new = build_transforms(config, train=True)(image)
    torch.manual_seed(3)
    old = _legacy_reference(config, True)(image)
    assert torch.equal(new, old)
    assert [type(s) for s in build_transforms(config, train=True).transforms] == [
        type(s) for s in _legacy_reference(config, True).transforms
    ]


@pytest.mark.parametrize("train", [False, True])
def test_head_crop_output_shape(train: bool) -> None:
    """Le mode head_crop sort un tenseur (3, 224, 224) décimal, même sur une image rectangulaire."""
    config = DataConfig(preprocessing="head_crop")
    tensor = build_transforms(config, train=train)(_ellipse_image())
    assert tensor.shape == (3, 224, 224)
    assert tensor.dtype == torch.float32


def test_head_crop_does_not_distort() -> None:
    """Une ellipse 160x60 sur une image 200x120 garde son rapport (2,67) en head_crop.

    En mode legacy, l'image est étirée en carré et le rapport change.
    """
    image = _ellipse_image(width=200, height=120, a=80, b=30)  # ellipse 160 x 60
    config = DataConfig(preprocessing="head_crop")
    gray = _denormalize(build_transforms(config, train=False)(image), config)
    assert _bright_ratio(gray) == pytest.approx(160 / 60, rel=0.05)

    legacy = DataConfig()
    legacy_gray = _denormalize(build_transforms(legacy, train=False)(image), legacy)
    assert _bright_ratio(legacy_gray) != pytest.approx(160 / 60, rel=0.05)


def test_head_crop_eval_is_deterministic() -> None:
    """Sans augmentation, le mode head_crop donne toujours le même tenseur."""
    transform = build_transforms(DataConfig(preprocessing="head_crop"), train=False)
    image = _noisy_image()
    assert torch.equal(transform(image), transform(image))


def test_head_crop_augmentation_is_seeded() -> None:
    """Même graine = même image augmentée ; des graines différentes, des images différentes."""
    transform = build_transforms(DataConfig(preprocessing="head_crop"), train=True)
    image = _noisy_image()
    outputs = []
    for seed in (1, 1, 2, 3):
        torch.manual_seed(seed)
        outputs.append(transform(image))
    assert torch.equal(outputs[0], outputs[1])
    assert not torch.equal(outputs[0], outputs[2]) or not torch.equal(outputs[0], outputs[3])


def test_head_crop_augmentation_steps() -> None:
    """La chaîne d'entraînement contient les augmentations demandées, dans le bon ordre."""
    steps = build_transforms(DataConfig(preprocessing="head_crop"), train=True).transforms
    names = [type(s).__name__ for s in steps]
    expected = [
        "RandomZoomOut",
        "RandomResizedCrop",
        "RandomHorizontalFlip",
        "RandomRotation",
        "ColorJitter",
        "RandomGamma",
        "RandomApply",  # flou
        "RandomApply",  # JPEG
        "ToDtype",
        "RandomApply",  # bruit
        "Normalize",
    ]
    assert [n for n in names if n not in ("Lambda", "ToImage")] == expected
    zoom = steps[names.index("RandomZoomOut")]
    assert list(zoom.side_range) == [1.0, 1.4] and zoom.p == 0.3
    crop = steps[names.index("RandomResizedCrop")]
    assert tuple(crop.scale) == (0.6, 1.0) and tuple(crop.ratio) == (0.8, 1.25)


def test_random_gamma_changes_intensity() -> None:
    """Un gamma > 1 assombrit une image 8 bits, un gamma < 1 l'éclaircit."""
    image = torch.full((3, 4, 4), 128, dtype=torch.uint8)
    assert RandomGamma((2.0, 2.0))(image).float().mean() < 128
    assert RandomGamma((0.5, 0.5))(image).float().mean() > 128
