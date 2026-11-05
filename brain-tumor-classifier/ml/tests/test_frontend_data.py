"""Tests de brain_mri.frontend_data (JSON des scores lu par la page « Résultats » du site)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from brain_mri.frontend_data import build_metrics_summary, write_metrics_json


def _evaluation() -> dict:
    """Construit un faux fichier d'évaluation à 2 classes, avec des champs en trop."""
    scores = {"precision": 0.9, "recall": 0.8, "f1": 0.85, "support": 10, "autre": 1}
    return {
        "accuracy": 0.85,
        "per_class": {"a": dict(scores), "b": dict(scores)},
        "confusion_matrix": [[8, 2], [1, 9]],
        "errors": [{"fichier": "x.jpg"}],
    }


def test_build_metrics_summary_keeps_only_expected_fields() -> None:
    """Le résumé contient exactement les champs de ``MetricsSummary``, rien de plus."""
    summary = build_metrics_summary(_evaluation())
    assert set(summary) == {"accuracy", "class_names", "per_class", "confusion_matrix"}
    assert summary["class_names"] == ["a", "b"]
    assert set(summary["per_class"]["a"]) == {"precision", "recall", "f1", "support"}
    assert summary["confusion_matrix"] == [[8, 2], [1, 9]]


def test_build_metrics_summary_rejects_bad_matrix() -> None:
    """Une matrice de mauvaise taille donne une erreur claire."""
    evaluation = _evaluation()
    evaluation["confusion_matrix"] = [[1]]
    with pytest.raises(ValueError, match="matrice"):
        build_metrics_summary(evaluation)


def test_write_metrics_json_writes_file(tmp_path: Path) -> None:
    """Le fichier est écrit (dossier créé) et relisible."""
    source = tmp_path / "evaluation.json"
    source.write_text(json.dumps(_evaluation()), encoding="utf-8")
    output = tmp_path / "sous" / "metrics.json"
    write_metrics_json(source, output)
    assert json.loads(output.read_text(encoding="utf-8"))["accuracy"] == 0.85


def test_write_metrics_json_missing_file(tmp_path: Path) -> None:
    """Un fichier d'évaluation absent donne une erreur claire."""
    with pytest.raises(FileNotFoundError):
        write_metrics_json(tmp_path / "absent.json", tmp_path / "metrics.json")
