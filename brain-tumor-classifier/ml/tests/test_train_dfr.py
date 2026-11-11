"""Tests de scripts/train_dfr.py : pire groupe et validation croisée sur données jouets."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import train_dfr  # noqa: E402  (le dossier scripts/ n'est pas un paquet)


def test_worst_group_accuracy_ignores_small_groups() -> None:
    """Le pire groupe est le minimum sur les groupes d'au moins 5 images ; les autres sont notés."""
    labels = np.array([0] * 10 + [1] * 10 + [2] * 3)
    is_512 = np.array([True] * 10 + [False] * 10 + [True] * 3)
    predictions = labels.copy()
    predictions[:2] = 1  # 2 erreurs sur 10 dans le groupe (0, 512)
    predictions[-3:] = 0  # le petit groupe (2, 512) de 3 images est entièrement faux
    worst, detail = train_dfr.worst_group_accuracy(labels, predictions, is_512)
    assert worst == 0.8
    assert detail["2/512"]["ignored"] is True
    assert detail["2/512"]["accuracy"] == 0.0
    assert detail["1/autre"]["accuracy"] == 1.0


def test_cross_validate_returns_out_of_fold_predictions_for_every_image() -> None:
    """Chaque image reçoit une prédiction hors pli pour chaque weight_decay de la grille."""
    from brain_mri.config import Config

    torch.manual_seed(0)
    n, dim = 120, 6
    labels = np.arange(n) % 4
    is_512 = (np.arange(n) // 4) % 2 == 0
    groups = np.array([str(i // 2) for i in range(n)])  # deux images par patient
    centers = torch.randn(4, dim) * 3
    x = centers[torch.from_numpy(labels)] + torch.randn(n, dim)
    train_x = x[:, None, :].repeat(1, 2, 1)  # 2 « vues » identiques
    config = Config.model_validate({"dinov2": {"epochs": 3, "patience": 3, "batch_size": 32}})
    result = train_dfr.cross_validate(
        train_x, x, labels, is_512, groups, x[:20], torch.from_numpy(labels[:20]), config, 4,
        (0.0, 0.1), 42,
    )  # fmt: skip
    assert len(result["folds"]) == train_dfr.N_FOLDS
    assert set(result["grid"]) == {"0.0", "0.1"}
    for entry in result["grid"].values():
        assert len(entry["predictions"]) == n
        assert min(entry["predictions"]) >= 0  # aucune image oubliée
