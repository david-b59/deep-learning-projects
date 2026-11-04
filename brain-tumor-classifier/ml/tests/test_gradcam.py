"""Tests de brain_mri.gradcam (faux jeu de données, poids aléatoires)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from brain_mri.config import Config, DataConfig, ModelConfig, TrainConfig
from brain_mri.evaluate import evaluate
from brain_mri.gradcam import (
    GALLERY_NAME,
    build_gallery,
    gradcam_captum,
    gradcam_manual,
    load_image,
    overlay_figure,
    predict_one,
    select_examples,
    upsample,
)
from brain_mri.model import build_model
from brain_mri.train import save_checkpoint

CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]


@pytest.fixture
def grad_config(config: DataConfig, tmp_path: Path) -> Config:
    """Configuration complète pour le faux jeu : sorties dans tmp_path."""
    training = TrainConfig(
        checkpoint_path=tmp_path / "models" / "best.pt",
        figures_dir=tmp_path / "figures",
        mlflow_dir=tmp_path / "mlruns",
    )
    return Config(data=config, model=ModelConfig(pretrained=False), training=training)


@pytest.fixture(scope="module")
def eval_model() -> torch.nn.Module:
    """Fournit un ResNet-50 à poids aléatoires, corps gelé, en mode évaluation."""
    model = build_model(4, ModelConfig(pretrained=False, freeze_backbone=True))
    return model.eval()


def test_manual_and_captum_give_the_same_map(eval_model: torch.nn.Module) -> None:
    """Le Grad-CAM à la main et celui de Captum donnent la même carte (après normalisation)."""
    torch.manual_seed(0)
    x = torch.randn(1, 3, 64, 64)
    manual = gradcam_manual(eval_model, x, target=2)
    captum = gradcam_captum(eval_model, x, target=2)
    assert manual.shape == captum.shape == (2, 2)  # 64 / 32 = 2 : layer4 réduit l'image de 32
    assert torch.allclose(manual, captum, atol=1e-5)


def test_heatmap_is_between_zero_and_one(eval_model: torch.nn.Module) -> None:
    """La carte normalisée reste dans [0, 1]."""
    heatmap = gradcam_captum(eval_model, torch.randn(1, 3, 64, 64), target=0)
    assert heatmap.min() >= 0 and heatmap.max() <= 1


def test_manual_hook_is_removed(eval_model: torch.nn.Module) -> None:
    """Le hook posé sur layer4 est retiré après le calcul."""
    gradcam_manual(eval_model, torch.randn(1, 3, 64, 64), target=1)
    assert len(eval_model.layer4._forward_hooks) == 0


def test_upsample_resizes_the_map() -> None:
    """La carte est agrandie à la taille de l'image."""
    big = upsample(torch.rand(7, 7), 224)
    assert big.shape == (224, 224)
    assert big.min() >= 0 and big.max() <= 1


def test_predict_one_returns_a_valid_class(eval_model: torch.nn.Module) -> None:
    """La classe prédite est un indice valide et la probabilité est entre 0 et 1."""
    index, probability = predict_one(eval_model, torch.randn(1, 3, 64, 64))
    assert 0 <= index < 4 and 0.0 <= probability <= 1.0


def test_load_image_shapes(grad_config: Config) -> None:
    """Le tenseur a la forme (1, 3, H, W) et l'image d'affichage (H, W, 3) en valeurs 0-1."""
    path = next((grad_config.data.raw_dir / "Testing" / "glioma").glob("*.jpg"))
    tensor, display = load_image(path, grad_config)
    size = grad_config.data.image_size
    assert tensor.shape == (1, 3, size, size)
    assert display.shape == (size, size, 3) and 0.0 <= display.min() and display.max() <= 1.0


def test_overlay_figure_writes_an_image(tmp_path: Path) -> None:
    """La galerie est enregistrée dans un PNG non vide, même avec un nombre impair d'images."""
    item = {
        "image": np.zeros((8, 8, 3), dtype=np.float32),
        "heatmap": np.ones((8, 8), dtype=np.float32),
        "title": "test",
        "kind": "ok",
    }
    path = tmp_path / "figs" / "gallery.png"
    overlay_figure([item, item, item], path)
    assert path.is_file() and path.stat().st_size > 0


def test_build_gallery_end_to_end(grad_config: Config) -> None:
    """Après evaluate, la galerie se construit à partir du modèle et des résultats."""
    model = build_model(4, ModelConfig(pretrained=False))
    save_checkpoint(grad_config.training.checkpoint_path, model, CLASS_NAMES, {})
    results = evaluate(grad_config)
    assert results["errors"]  # un modèle aléatoire se trompe : il y a des erreurs à montrer
    path = build_gallery(grad_config)
    assert path.name == GALLERY_NAME and path.stat().st_size > 0


def test_build_gallery_without_evaluation_raises(grad_config: Config) -> None:
    """Sans evaluation.json, l'erreur est claire."""
    with pytest.raises(FileNotFoundError):
        build_gallery(grad_config)


def test_evaluation_json_is_readable_by_gallery(grad_config: Config) -> None:
    """Le fichier produit par evaluate contient les clés dont la galerie a besoin."""
    model = build_model(4, ModelConfig(pretrained=False))
    save_checkpoint(grad_config.training.checkpoint_path, model, CLASS_NAMES, {})
    evaluate(grad_config)
    saved = json.loads((grad_config.training.figures_dir / "evaluation.json").read_text("utf-8"))
    assert {"errors", "top_confusions"} <= saved.keys()


def test_select_examples_picks_good_then_errors(grad_config: Config) -> None:
    """La sélection donne une bonne prédiction par classe, puis les erreurs demandées."""
    raw = grad_config.data.raw_dir
    samples = [
        (str(path), label)
        for label, name in enumerate(CLASS_NAMES)
        for path in sorted((raw / "Testing" / name).glob("*.jpg"))
    ]
    errors = [
        {"file": "Testing/glioma/img_1.jpg", "true": "glioma", "predicted": "notumor"},
        {"file": "Testing/glioma/img_2.jpg", "true": "glioma", "predicted": "notumor"},
        {"file": "Testing/glioma/img_3.jpg", "true": "glioma", "predicted": "notumor"},
        {"file": "Testing/notumor/img_1.jpg", "true": "notumor", "predicted": "glioma"},
    ]
    confusions = [
        {"true": "glioma", "predicted": "notumor", "count": 3},
        {"true": "notumor", "predicted": "glioma", "count": 1},
    ]
    results = {"errors": errors, "top_confusions": confusions}

    chosen = select_examples(samples, CLASS_NAMES, results, grad_config, per_confusion=2)
    good = [c for c in chosen if c["kind"] == "ok"]
    wrong = [c for c in chosen if c["kind"] == "erreur"]
    assert {c["true"] for c in good} == set(CLASS_NAMES)  # une par classe
    assert len(wrong) == 3  # 2 (sur 3 disponibles) + 1
    wrong_files = {e["file"] for e in errors}
    assert all(c["path"].relative_to(raw).as_posix() not in wrong_files for c in good)
    # Même graine, même choix.
    assert chosen == select_examples(samples, CLASS_NAMES, results, grad_config, per_confusion=2)
