"""Tests de brain_mri.leakage (faux embeddings, faux modèle, fausses images)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from brain_mri.config import DataConfig
from brain_mri.leakage import (
    BIN_LABELS,
    accuracy_summary,
    analyze_invariance,
    analyze_kaggle_leak,
    assign_bins,
    predict_paths,
    sample_indices,
    select_flips,
    select_pairs_by_bin,
)
from brain_mri.transforms import build_transforms

CLASS_NAMES = ["glioma", "notumor"]


class ConstantModel(nn.Module):
    """Faux modèle qui prédit toujours la classe 1."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Renvoie des scores favorisant la classe 1 pour chaque image du lot."""
        out = torch.zeros(x.shape[0], 2)
        out[:, 1] = 5.0
        return out


def test_assign_bins() -> None:
    """Les bornes basses sont incluses dans leur tranche."""
    cos = np.array([0.5, 0.90, 0.939, 0.94, 0.96, 0.979, 0.98, 1.0])
    assert assign_bins(cos).tolist() == [0, 1, 1, 2, 3, 3, 4, 4]
    assert len(BIN_LABELS) == 5


def test_accuracy_summary() -> None:
    """Groupe vide : exactitude None ; sinon la moyenne du groupe."""
    ok = np.array([True, False, True, True])
    assert accuracy_summary(ok, np.array([True, True, False, False])) == {"n": 2, "accuracy": 0.5}
    assert accuracy_summary(ok, np.zeros(4, dtype=bool)) == {"n": 0, "accuracy": None}


def test_analyze_kaggle_leak() -> None:
    """Cas calculé à la main : jumelles, même classe, exactitude avec et sans jumelle."""
    cos = np.array([0.99, 0.95, 0.50, 0.97])
    same = np.array([True, False, True, True])
    labels = np.array([0, 0, 1, 1])
    ok = np.array([True, False, True, True])
    res = analyze_kaggle_leak(cos, same, labels, ok, CLASS_NAMES)
    o = res["overall"]
    assert o["n"] == 4 and o["with_twin"] == 3
    assert o["twin_same_class_rate"] == pytest.approx(2 / 3)
    assert o["with_twin_accuracy"] == {"n": 3, "accuracy": pytest.approx(2 / 3)}
    assert o["without_twin_accuracy"] == {"n": 1, "accuracy": 1.0}
    glioma = res["per_class"]["glioma"]
    assert glioma["with_twin"] == 2 and glioma["twin_same_class_rate"] == pytest.approx(0.5)
    assert res["by_cosine_bin"]["<0,90"]["n"] == 1
    assert res["by_cosine_bin"]["0,94-0,96"]["n"] == 1
    assert res["by_cosine_bin"][">=0,98"]["per_class"]["glioma"]["accuracy"] == 1.0


def _invariance_inputs() -> tuple:
    """Cinq images PMRAM : trois copies sûres de même classe, une jumelle d'une autre classe."""
    cos = np.array([0.99, 0.99, 0.985, 0.99, 0.95])
    same = np.array([True, True, True, False, True])
    labels = np.array([0, 0, 1, 1, 0])
    copy_pred = np.array([0, 1, 1, 0, 0])
    twin_pred = np.array([0, 0, 1, 1, 1])
    return cos, same, labels, copy_pred, twin_pred


def test_analyze_invariance() -> None:
    """Les copies sûres, les bascules et les sens de bascule sont comptés correctement."""
    res = analyze_invariance(*_invariance_inputs(), CLASS_NAMES)
    safe = res["safe"]
    assert safe["n"] == 3  # l'image 3 (autre classe) et l'image 4 (cos 0,95) sont écartées
    assert safe["twin_accuracy"] == pytest.approx(1.0)
    assert safe["copy_accuracy"] == pytest.approx(2 / 3)
    assert safe["flips"] == 1 and safe["right_to_wrong"] == 1 and safe["wrong_to_right"] == 0
    assert res["safe_per_class"]["glioma"]["flips"] == 1
    assert res["safe_per_class"]["notumor"]["n"] == 1
    mid = res["by_cosine_bin"]["0,94-0,96"]
    assert mid["n_all"] == 1 and mid["same_class_twin"]["wrong_to_right"] == 1
    assert res["by_cosine_bin"][">=0,98"]["n_all"] == 4


def test_selection_helpers() -> None:
    """Le tirage est déterministe, borné, et ne choisit que de vraies bascules."""
    assert sample_indices([5, 1, 9, 3], 2) == sample_indices([5, 1, 9, 3], 2)
    assert sample_indices([1, 2], 10) == [1, 2]
    cos, same, _, copy_pred, twin_pred = _invariance_inputs()
    assert select_flips(cos, same, copy_pred, twin_pred, count=5) == [1]
    cos = np.array([0.95, 0.95, 0.99, 0.99, 0.10])
    labels = np.array([0, 1, 0, 0, 0])
    chosen = select_pairs_by_bin(cos, labels, [2, 4], [0, 1], per_cell=1)
    assert chosen == [(0, 2, 0), (1, 2, 1), (2, 4, 0)]  # aucun notumor dans la tranche >=0,98


def test_predict_paths(tmp_path: Path) -> None:
    """Le faux modèle prédit la classe 1 pour chaque fichier, dans l'ordre et par lots."""
    paths = []
    for i in range(3):
        path = tmp_path / f"{i}.png"
        Image.fromarray(np.full((20, 20, 3), i * 50, dtype=np.uint8)).save(path)
        paths.append(path)
    transform = build_transforms(DataConfig(image_size=16, batch_size=2), train=False)
    result = predict_paths(ConstantModel(), paths, transform, torch.device("cpu"), batch_size=2)
    assert result.tolist() == [1, 1, 1]
    assert predict_paths(ConstantModel(), [], transform, torch.device("cpu")).tolist() == []


def test_bootstrap_accuracy_interval_contains_accuracy_and_is_reproducible() -> None:
    """L'IC encadre l'exactitude, est reproductible (même graine) et gère le cas vide."""
    from brain_mri.leakage import bootstrap_accuracy

    correct = np.array([True] * 80 + [False] * 20)
    result = bootstrap_accuracy(correct, draws=500, seed=42)
    assert result["n"] == 100
    assert result["accuracy"] == 0.8
    assert result["ci_low"] < 0.8 < result["ci_high"]
    assert result == bootstrap_accuracy(correct, draws=500, seed=42)
    assert bootstrap_accuracy(np.array([], dtype=bool))["accuracy"] is None
