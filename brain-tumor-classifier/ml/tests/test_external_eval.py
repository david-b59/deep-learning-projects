"""Tests de brain_mri.external_eval (faux manifeste, faux modèle, fausses images)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from brain_mri.config import DataConfig
from brain_mri.external_eval import (
    ExternalDataset,
    evaluate_subset,
    resolve_path,
    select_subsets,
    summarize,
)
from brain_mri.transforms import build_transforms

CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]


class ConstantModel(nn.Module):
    """Faux modèle qui prédit toujours la même classe, avec un score très net."""

    def __init__(self, index: int) -> None:
        """Mémorise l'indice de la classe prédite."""
        super().__init__()
        self.index = index

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Renvoie des scores favorisant ``index`` pour chaque image du lot."""
        out = torch.zeros(x.shape[0], 4)
        out[:, self.index] = 10.0
        return out


def _entries() -> list[dict]:
    """Quatre entrées : gardée, proche à 4 bits, doublon à 1 bit, exacte (0 bit)."""
    return [
        {"file": "glioma/a.jpg", "class": "glioma", "status": "kept", "distance": None},
        {"file": "notumor/b.jpg", "class": "notumor", "status": "excluded_near", "distance": 4},
        {"file": "glioma/c.jpg", "class": "glioma", "status": "excluded_near", "distance": 1},
        {
            "file": "pituitary/d.jpg",
            "class": "pituitary",
            "status": "excluded_exact",
            "distance": 0,
        },
    ]


def test_select_subsets() -> None:
    """Les trois sous-ensembles suivent les règles du statut et de la distance."""
    subsets = select_subsets(_entries())
    assert [e["file"] for e in subsets["strict"]] == ["glioma/a.jpg"]
    assert [e["file"] for e in subsets["seuil_2"]] == ["glioma/a.jpg", "notumor/b.jpg"]
    assert [e["file"] for e in subsets["doublons"]] == ["glioma/c.jpg", "pituitary/d.jpg"]


def test_resolve_path(tmp_path: Path) -> None:
    """Une image gardée est dans son dossier, une écartée dans _excluded."""
    kept, near = _entries()[0], _entries()[1]
    assert resolve_path(tmp_path, kept) == tmp_path / "glioma/a.jpg"
    assert resolve_path(tmp_path, near) == tmp_path / "_excluded" / "notumor/b.jpg"


def test_resolve_path_fallback(tmp_path: Path) -> None:
    """Une image écartée mais restée à sa place (cas BDNeuro) est retrouvée, et inversement."""
    gray = {"file": "glioma/g.jpg", "class": "glioma", "status": "gray_zone", "distance": None}
    kept = {"file": "notumor/k.jpg", "class": "notumor", "status": "kept", "distance": None}
    (tmp_path / "glioma").mkdir()
    (tmp_path / "glioma" / "g.jpg").write_bytes(b"x")
    (tmp_path / "_excluded" / "notumor").mkdir(parents=True)
    (tmp_path / "_excluded" / "notumor" / "k.jpg").write_bytes(b"x")
    assert resolve_path(tmp_path, gray) == tmp_path / "glioma" / "g.jpg"  # pas déplacée
    assert resolve_path(tmp_path, kept) == tmp_path / "_excluded" / "notumor" / "k.jpg"
    missing = {"file": "glioma/z.jpg", "class": "glioma", "status": "excluded_copy"}
    assert resolve_path(tmp_path, missing) == tmp_path / "_excluded" / "glioma" / "z.jpg"


def test_summarize_known_case() -> None:
    """Sur un cas calculé à la main, les indicateurs sont exacts."""
    y_true = [0, 0, 2, 2, 3]
    y_pred = [0, 2, 2, 0, 3]  # un glioma pris pour notumor, un notumor pris pour glioma
    conf = [0.9, 0.6, 0.8, 0.5, 1.0]
    res = summarize(y_true, y_pred, conf, CLASS_NAMES)
    assert res["accuracy"] == pytest.approx(3 / 5)
    assert res["per_class_n"]["glioma"] == 2
    assert res["recall"]["glioma"] == pytest.approx(0.5)
    assert res["tumor_predicted_notumor"] == {"count": 1, "total": 3, "rate": pytest.approx(1 / 3)}
    assert res["notumor_predicted_tumor"] == {"count": 1, "total": 2}
    assert res["mean_confidence_correct"] == pytest.approx((0.9 + 0.8 + 1.0) / 3)
    assert res["mean_confidence_wrong"] == pytest.approx(0.55)


def test_summarize_empty_and_no_errors() -> None:
    """Liste vide : seulement n ; aucune erreur : confiance des mauvaises = None."""
    assert summarize([], [], [], CLASS_NAMES) == {"n": 0}
    res = summarize([0, 1], [0, 1], [0.7, 0.9], CLASS_NAMES)
    assert res["mean_confidence_wrong"] is None
    assert res["tumor_predicted_notumor"]["count"] == 0


def test_evaluate_subset_with_fake_images(tmp_path: Path) -> None:
    """Le chargement lit kept et _excluded, et un modèle constant donne le bon rappel."""
    entries = _entries()
    for entry in entries:
        path = resolve_path(tmp_path, entry)
        path.parent.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(0)
        Image.fromarray(rng.integers(0, 256, (20, 20, 3), dtype=np.uint8)).save(path)
    config = DataConfig(image_size=16, batch_size=2)
    transform = build_transforms(config, train=False)

    dataset = ExternalDataset(tmp_path, entries, CLASS_NAMES, transform)
    assert len(dataset) == 4
    assert dataset[0][0].shape == (3, 16, 16)

    cpu = torch.device("cpu")
    res = evaluate_subset(ConstantModel(0), tmp_path, entries, CLASS_NAMES, transform, cpu, 2)
    assert res["n"] == 4
    assert res["recall"]["glioma"] == pytest.approx(1.0)  # modèle « toujours glioma »
    assert res["accuracy"] == pytest.approx(2 / 4)
    assert res["notumor_predicted_tumor"] == {"count": 1, "total": 1}
    assert evaluate_subset(ConstantModel(0), tmp_path, [], CLASS_NAMES, transform, cpu) == {"n": 0}
