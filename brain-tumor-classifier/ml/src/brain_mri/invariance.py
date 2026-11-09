"""Mesure de l'invariance du modèle : taux de bascule sous transformations (étape 11b).

Une « bascule » est une image pour laquelle la prédiction change quand on transforme l'image
AVANT le prétraitement du modèle. Deux mesures :

    1. **Transformations fixes** (:data:`TRANSFORMATIONS`) : miroir, bandes noires, zoom,
       recadrage décentré, étirement, JPEG. Le taux de bascule est donné au total et par
       classe, avec la décomposition ``right_to_wrong`` (juste -> faux) et ``wrong_to_right``
       (faux -> juste).
    2. **Paires PMRAM / Kaggle** : voir :func:`analyze_pmram_by_twin_part`, qui réutilise
       :func:`brain_mri.leakage.analyze_invariance` et sépare, quand un manifeste de découpage
       est fourni, les jumelles Kaggle placées en entraînement de celles placées en test.

Le module ne contient que de la logique pure et des fonctions de prédiction ;
``scripts/evaluate_invariance.py`` fournit les fichiers, le modèle et les embeddings.
"""

from __future__ import annotations

import io
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch import nn
from torchvision.transforms import v2

from brain_mri.leakage import analyze_invariance, predict_paths

BANDS_FRACTION = 0.25  # bandes noires ajoutées de chaque côté, en part de la dimension
OFFCENTER_KEEP = 0.90  # part de l'image gardée par le recadrage décentré
STRETCH = 0.20  # étirement horizontal de ± 20 %
JPEG_QUALITY = 50  # qualité JPEG du test de compression


def mirror(image: Image.Image) -> Image.Image:
    """Miroir horizontal."""
    return ImageOps.mirror(image)


def add_black_bands(image: Image.Image) -> Image.Image:
    """Ajoute 25 % de bande noire de chaque côté (l'image occupe alors 2/3 de la largeur)."""
    width, height = image.size
    pad_x, pad_y = round(width * BANDS_FRACTION), round(height * BANDS_FRACTION)
    canvas = Image.new(image.mode, (width + 2 * pad_x, height + 2 * pad_y), 0)
    canvas.paste(image, (pad_x, pad_y))
    return canvas


def zoom(image: Image.Image, factor: float) -> Image.Image:
    """Zoome d'un facteur ``factor`` en gardant la taille d'origine.

    Un facteur > 1 recadre le centre puis agrandit ; un facteur < 1 réduit l'image et la
    centre sur un fond noir.

    Args:
        image: Image d'origine.
        factor: Facteur de zoom (par ex. 0,85 ou 1,15).

    Returns:
        Une image de même taille que l'originale.
    """
    width, height = image.size
    if factor >= 1:
        crop_w, crop_h = round(width / factor), round(height / factor)
        left, top = (width - crop_w) // 2, (height - crop_h) // 2
        return image.crop((left, top, left + crop_w, top + crop_h)).resize((width, height))
    small = image.resize((round(width * factor), round(height * factor)))
    canvas = Image.new(image.mode, (width, height), 0)
    canvas.paste(small, ((width - small.width) // 2, (height - small.height) // 2))
    return canvas


def offcenter_crop(image: Image.Image) -> Image.Image:
    """Garde 90 % de l'image, ancrés en bas à droite (décentrés), remis à la taille d'origine."""
    width, height = image.size
    box = (
        round(width * (1 - OFFCENTER_KEEP)),
        round(height * (1 - OFFCENTER_KEEP)),
        width,
        height,
    )
    return image.crop(box).resize((width, height))


def stretch_horizontal(image: Image.Image, factor: float) -> Image.Image:
    """Étire l'image en largeur d'un facteur ``factor`` (hauteur inchangée)."""
    return image.resize((round(image.width * factor), image.height))


def jpeg(image: Image.Image) -> Image.Image:
    """Recompresse l'image en JPEG de qualité 50 puis la relit."""
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=JPEG_QUALITY)
    buffer.seek(0)
    with Image.open(buffer) as compressed:
        return compressed.convert("RGB")


# Transformations fixes, dans l'ordre d'affichage : nom -> fonction image PIL -> image PIL.
TRANSFORMATIONS: dict[str, Callable[[Image.Image], Image.Image]] = {
    "miroir": mirror,
    "bandes_noires_25": add_black_bands,
    "zoom_0.85": lambda image: zoom(image, 0.85),
    "zoom_1.15": lambda image: zoom(image, 1.15),
    "recadrage_decentre_90": offcenter_crop,
    "etirement_+20": lambda image: stretch_horizontal(image, 1 + STRETCH),
    "etirement_-20": lambda image: stretch_horizontal(image, 1 - STRETCH),
    "jpeg_50": jpeg,
}


def flip_summary(labels: np.ndarray, base_pred: np.ndarray, new_pred: np.ndarray) -> dict:
    """Résume les bascules d'un groupe d'images.

    Args:
        labels: Vraie classe de chaque image.
        base_pred: Prédiction sur l'image d'origine.
        new_pred: Prédiction sur l'image transformée.

    Returns:
        ``n``, ``flips`` (prédiction différente), ``flip_rate`` (None si groupe vide),
        ``right_to_wrong`` (juste avant, faux après) et ``wrong_to_right``.
    """
    n = len(labels)
    base_ok, new_ok = base_pred == labels, new_pred == labels
    flips = int((base_pred != new_pred).sum())
    return {
        "n": n,
        "flips": flips,
        "flip_rate": flips / n if n else None,
        "right_to_wrong": int((base_ok & ~new_ok).sum()),
        "wrong_to_right": int((~base_ok & new_ok).sum()),
    }


def analyze_transformations(
    labels: np.ndarray,
    base_pred: np.ndarray,
    new_preds: dict[str, np.ndarray],
    class_names: list[str],
) -> dict:
    """Taux de bascule de chaque transformation, au total et par classe.

    Args:
        labels: Vraie classe (indice) de chaque image.
        base_pred: Prédictions sur les images d'origine.
        new_preds: Nom de la transformation -> prédictions sur les images transformées.
        class_names: Noms des classes.

    Returns:
        ``{"n", "base_accuracy", "transformations": {nom: {"overall", "accuracy", "per_class"}}}``.
    """

    def per_class(preds: np.ndarray) -> dict:
        """Bascules de chaque classe (selon la vraie classe)."""
        return {
            name: flip_summary(labels[labels == i], base_pred[labels == i], preds[labels == i])
            for i, name in enumerate(class_names)
        }

    return {
        "n": len(labels),
        "base_accuracy": float((base_pred == labels).mean()),
        "transformations": {
            name: {
                "overall": flip_summary(labels, base_pred, preds),
                "accuracy": float((preds == labels).mean()),
                "per_class": per_class(preds),
            }
            for name, preds in new_preds.items()
        },
    }


def _chain(
    first: Callable[[Image.Image], Image.Image], second: Callable[[Image.Image], torch.Tensor]
) -> Callable[[Image.Image], torch.Tensor]:
    """Compose deux étapes : ``first`` (image -> image) puis ``second`` (image -> tenseur)."""

    def chained(image: Image.Image) -> torch.Tensor:
        """Applique ``first`` puis ``second``."""
        return second(first(image))

    return chained


def measure_transformations(
    model: nn.Module,
    paths: Sequence[Path],
    labels: np.ndarray,
    transform: v2.Compose,
    device: torch.device,
    class_names: list[str],
    batch_size: int = 32,
    transformations: dict[str, Callable[[Image.Image], Image.Image]] | None = None,
) -> dict:
    """Mesure le taux de bascule du modèle pour chaque transformation fixe.

    La transformation est appliquée à l'image RGB AVANT ``transform`` (le prétraitement du
    modèle : ``legacy`` ou ``head_crop``).

    Args:
        model: Le modèle entraîné.
        paths: Chemins des images.
        labels: Vraie classe (indice) de chaque image.
        transform: Prétraitement du modèle (``build_transforms(train=False)``).
        device: CPU ou GPU.
        class_names: Noms des classes.
        batch_size: Taille des lots.
        transformations: Transformations à mesurer (par défaut :data:`TRANSFORMATIONS`).

    Returns:
        Le résultat de :func:`analyze_transformations`.
    """
    transformations = TRANSFORMATIONS if transformations is None else transformations
    base_pred = predict_paths(model, paths, transform, device, batch_size)
    new_preds = {
        name: predict_paths(model, paths, _chain(function, transform), device, batch_size)
        for name, function in transformations.items()
    }
    return analyze_transformations(labels, base_pred, new_preds, class_names)


def analyze_pmram_by_twin_part(
    cosines: np.ndarray,
    twin_same_class: np.ndarray,
    labels: np.ndarray,
    copy_pred: np.ndarray,
    twin_pred: np.ndarray,
    twin_part: np.ndarray | None,
    class_names: list[str],
) -> dict:
    """Bascules PMRAM (copies sûres), au total puis selon la partie de la jumelle Kaggle.

    Quand un manifeste de découpage est utilisé, une copie PMRAM dont la jumelle est en
    ``train`` n'est pas un test indépendant (le modèle a vu la jumelle) ; celle dont la
    jumelle est en ``test`` ou ``val`` l'est davantage. On sépare donc les deux.

    Args:
        cosines: Cosinus de chaque image PMRAM avec sa plus proche image Kaggle.
        twin_same_class: Booléens « la jumelle est de la même classe ».
        labels: Vraie classe de chaque image PMRAM.
        copy_pred: Classe prédite sur la copie PMRAM.
        twin_pred: Classe prédite sur la jumelle Kaggle.
        twin_part: Partie (``train``/``val``/``test``) de chaque jumelle, ou None sans manifeste.
        class_names: Noms des classes.

    Returns:
        ``{"all": résultat de analyze_invariance}`` puis, si ``twin_part`` est donné, une clé
        par partie présente (``twin_in_train``, ``twin_in_val``, ``twin_in_test``).
    """
    report = {
        "all": analyze_invariance(
            cosines, twin_same_class, labels, copy_pred, twin_pred, class_names
        )
    }
    if twin_part is not None:
        for part in ("train", "val", "test"):
            mask = twin_part == part
            if mask.any():
                report[f"twin_in_{part}"] = analyze_invariance(
                    cosines[mask],
                    twin_same_class[mask],
                    labels[mask],
                    copy_pred[mask],
                    twin_pred[mask],
                    class_names,
                )
    return report
