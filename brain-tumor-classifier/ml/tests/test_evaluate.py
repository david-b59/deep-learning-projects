"""Tests de brain_mri.evaluate (faux jeu de données, poids aléatoires)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from brain_mri.config import Config, DataConfig, ModelConfig, TrainConfig
from brain_mri.evaluate import (
    compute_metrics,
    evaluate,
    list_errors,
    plot_confusion_matrix,
    predict,
)
from brain_mri.model import build_model
from brain_mri.train import save_checkpoint

CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]


@pytest.fixture
def eval_config(config: DataConfig, tmp_path: Path) -> Config:
    """Configuration complète pour le faux jeu : sorties dans tmp_path."""
    training = TrainConfig(
        checkpoint_path=tmp_path / "models" / "best.pt",
        figures_dir=tmp_path / "figures",
        mlflow_dir=tmp_path / "mlruns",
    )
    return Config(data=config, model=ModelConfig(pretrained=False), training=training)


def test_compute_metrics_known_case() -> None:
    """Sur un cas calculé à la main, accuracy, matrice et confusions sont exacts."""
    y_true = [0, 0, 0, 1, 1, 2, 2, 3]
    y_pred = [0, 0, 1, 1, 1, 2, 0, 3]  # 2 erreurs : un 0 pris pour 1, un 2 pris pour 0
    metrics = compute_metrics(y_true, y_pred, CLASS_NAMES)
    assert metrics["accuracy"] == pytest.approx(6 / 8)
    assert metrics["confusion_matrix"][0] == [2, 1, 0, 0]
    assert metrics["per_class"]["glioma"]["recall"] == pytest.approx(2 / 3)
    assert metrics["per_class"]["glioma"]["precision"] == pytest.approx(2 / 3)
    assert metrics["per_class"]["glioma"]["support"] == 3
    assert {c["count"] for c in metrics["top_confusions"]} == {1}
    assert len(metrics["top_confusions"]) == 2


def test_compute_metrics_handles_class_never_predicted() -> None:
    """Une classe jamais prédite donne une précision de 0, sans erreur."""
    metrics = compute_metrics([0, 1, 2, 3], [0, 0, 2, 3], CLASS_NAMES)
    assert metrics["per_class"]["meningioma"]["precision"] == 0.0
    assert metrics["per_class"]["meningioma"]["recall"] == 0.0


def test_list_errors_keeps_only_wrong_predictions() -> None:
    """Seules les images mal classées sont listées, avec leurs noms de classes."""
    errors = list_errors(["a.jpg", "b.jpg", "c.jpg"], [0, 1, 2], [0, 2, 2], CLASS_NAMES)
    assert errors == [{"file": "b.jpg", "true": "meningioma", "predicted": "notumor"}]


def test_plot_confusion_matrix_writes_an_image(tmp_path: Path) -> None:
    """La matrice de confusion est enregistrée dans un PNG non vide."""
    path = tmp_path / "figs" / "cm.png"
    plot_confusion_matrix([[2, 0], [1, 3]], ["a", "b"], path)
    assert path.is_file() and path.stat().st_size > 0


def test_predict_returns_one_prediction_per_image(config: DataConfig, eval_config: Config) -> None:
    """``predict`` renvoie autant de prédictions que d'images de test, dans les classes valides."""
    from brain_mri.data import build_dataloaders

    bundle = build_dataloaders(config)
    model = build_model(4, ModelConfig(pretrained=False))
    y_true, y_pred = predict(model, bundle.test_loader, torch.device("cpu"))
    assert len(y_true) == len(y_pred) == bundle.sizes["test"]
    assert set(y_pred) <= {0, 1, 2, 3}


def test_evaluate_writes_results(eval_config: Config) -> None:
    """Une évaluation complète écrit le JSON et la figure, et les chiffres sont cohérents."""
    model = build_model(4, ModelConfig(pretrained=False))
    save_checkpoint(
        eval_config.training.checkpoint_path, model, CLASS_NAMES, {"val_loss": 0.5, "epoch": 1}
    )
    results = evaluate(eval_config)

    total = sum(map(sum, results["confusion_matrix"]))
    assert 0.0 <= results["accuracy"] <= 1.0
    assert len(results["errors"]) == total - sum(
        results["confusion_matrix"][i][i] for i in range(4)
    )
    assert results["checkpoint"]["epoch_in_phase"] == 1
    figures = eval_config.training.figures_dir
    assert (figures / "confusion_matrix.png").is_file()
    saved = json.loads((figures / "evaluation.json").read_text(encoding="utf-8"))
    assert saved["accuracy"] == results["accuracy"]
    assert not any(Path(e["file"]).is_absolute() for e in saved["errors"])  # chemins relatifs


def test_evaluate_without_checkpoint_raises(eval_config: Config) -> None:
    """Sans fichier de modèle, l'erreur est claire."""
    with pytest.raises(FileNotFoundError):
        evaluate(eval_config)


def test_evaluate_rejects_mismatched_classes(eval_config: Config) -> None:
    """Un modèle entraîné sur d'autres classes est refusé."""
    model = build_model(4, ModelConfig(pretrained=False))
    save_checkpoint(eval_config.training.checkpoint_path, model, ["a", "b", "c", "d"], {})
    with pytest.raises(ValueError):
        evaluate(eval_config)
