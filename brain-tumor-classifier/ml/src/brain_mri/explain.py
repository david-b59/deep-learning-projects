"""Grad-CAM avec Captum : calcul de la carte de chaleur, sans dépendance d'entraînement.

Module léger (torch, Captum et NumPy seulement), utilisé par la prédiction
(:mod:`brain_mri.predict`) et par la galerie (:mod:`brain_mri.gradcam`).
"""

from __future__ import annotations

import numpy as np
import torch
from captum.attr import LayerAttribution, LayerGradCam
from torch import nn


def _normalize(heatmap: torch.Tensor) -> torch.Tensor:
    """Ramène une carte entre 0 et 1 en la divisant par son maximum (carte nulle : inchangée).

    Args:
        heatmap: Carte à valeurs positives.

    Returns:
        La carte normalisée.
    """
    peak = heatmap.max()
    return heatmap / peak if peak > 0 else heatmap


def gradcam_captum(
    model: nn.Module, x: torch.Tensor, target: int, layer: nn.Module | None = None
) -> torch.Tensor:
    """Grad-CAM avec Captum (``LayerGradCam``).

    Args:
        model: Le modèle (en ``eval()``).
        x: Une image prétraitée, forme ``(1, 3, H, W)``.
        target: Indice de la classe à expliquer.
        layer: Couche à observer (par défaut ``model.layer4``).

    Returns:
        La carte, forme ``(h, w)``, valeurs entre 0 et 1.
    """
    layer = layer if layer is not None else model.layer4
    attribution = LayerGradCam(model, layer).attribute(x, target=target, relu_attributions=True)
    return _normalize(attribution.detach()[0, 0])  # (1, 1, 7, 7) -> (7, 7)


def upsample(heatmap: torch.Tensor, size: int) -> np.ndarray:
    """Agrandit une carte 7 x 7 à la taille de l'image (interpolation bilinéaire).

    Args:
        heatmap: Carte ``(h, w)``.
        size: Côté de l'image, en pixels.

    Returns:
        La carte ``(size, size)`` en NumPy, bornée entre 0 et 1.
    """
    big = LayerAttribution.interpolate(
        heatmap[None, None], (size, size), interpolate_mode="bilinear"
    )
    return big[0, 0].clamp(0, 1).numpy()
