"""Tests de l'export ONNX sur un modèle aléatoire (images 32 x 32, aucune donnée réelle)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest
import torch
import torch.nn.functional as F

from brain_mri.config import ModelConfig
from brain_mri.explain import gradcam_captum
from brain_mri.export_onnx import (
    cam_correlation,
    decide,
    export_onnx,
    normalize_cam,
    quantize_onnx,
)
from brain_mri.model import build_model

SIZE = 32  # côté des images de test : layer4 donne alors une carte 1 x 1
N_CLASSES = 4


@pytest.fixture(scope="module")
def random_model() -> torch.nn.Module:
    """ResNet-50 à poids aléatoires (graine fixe), en mode évaluation."""
    torch.manual_seed(0)
    model = build_model(N_CLASSES, ModelConfig(pretrained=False, freeze_backbone=False))
    return model.eval()


@pytest.fixture(scope="module")
def onnx_path(random_model: torch.nn.Module, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Exporte le modèle aléatoire une seule fois pour tous les tests du fichier."""
    return export_onnx(random_model, tmp_path_factory.mktemp("onnx") / "model.onnx", SIZE)


def _random_batch(seed: int = 1) -> np.ndarray:
    """Une image aléatoire normalisée, forme (1, 3, SIZE, SIZE), float32."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((1, 3, SIZE, SIZE)).astype(np.float32)


@pytest.mark.slow
def test_export_and_shapes(onnx_path: Path) -> None:
    """Le fichier existe, onnxruntime le charge et les sorties ont les bonnes formes."""
    assert onnx_path.is_file()
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    assert [i.name for i in session.get_inputs()] == ["image"]
    probs, cams = session.run(["probs", "cams"], {"image": _random_batch()})
    assert probs.shape == (1, N_CLASSES)
    assert cams.shape == (1, N_CLASSES, 1, 1)  # 32 pixels / 32 = carte 1 x 1


@pytest.mark.slow
def test_probabilities_sum_to_one_and_cams_positive(onnx_path: Path) -> None:
    """Les probabilités somment à 1 et les cartes sont toutes positives ou nulles (ReLU)."""
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    for seed in range(5):
        probs, cams = session.run(["probs", "cams"], {"image": _random_batch(seed)})
        assert probs.sum() == pytest.approx(1.0, abs=1e-5)
        assert (cams >= 0).all()


@pytest.mark.slow
def test_onnx_matches_pytorch(random_model: torch.nn.Module, onnx_path: Path) -> None:
    """ONNX et PyTorch donnent les mêmes probabilités (1e-4) et la même carte normalisée."""
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    x = _random_batch(7)
    probs, cams = session.run(["probs", "cams"], {"image": x})
    with torch.no_grad():
        expected = F.softmax(random_model(torch.from_numpy(x)), dim=1)[0].numpy()
    assert np.abs(probs[0] - expected).max() <= 1e-4
    target = int(expected.argmax())
    reference = gradcam_captum(random_model, torch.from_numpy(x), target).numpy()
    assert np.abs(normalize_cam(cams[0, target]) - reference).max() <= 1e-4


@pytest.mark.slow
def test_quantize_produces_loadable_smaller_file(onnx_path: Path, tmp_path: Path) -> None:
    """La quantification int8 produit un fichier plus petit que charge onnxruntime."""
    calibration = [_random_batch(seed) for seed in range(4)]
    int8_path = quantize_onnx(onnx_path, tmp_path / "model.int8.onnx", calibration)
    assert int8_path.stat().st_size < onnx_path.stat().st_size
    session = ort.InferenceSession(str(int8_path), providers=["CPUExecutionProvider"])
    probs, cams = session.run(["probs", "cams"], {"image": _random_batch()})
    assert probs.shape == (1, N_CLASSES)
    assert cams.shape == (1, N_CLASSES, 1, 1)


def test_normalize_cam() -> None:
    """La carte est divisée par son maximum ; une carte nulle reste inchangée."""
    assert normalize_cam(np.array([1.0, 2.0, 4.0])).max() == 1.0
    assert normalize_cam(np.zeros(3)).tolist() == [0.0, 0.0, 0.0]


def test_cam_correlation() -> None:
    """Corrélation de 1 pour deux cartes proportionnelles, cas constant géré sans erreur."""
    a = np.array([[0.0, 1.0], [2.0, 3.0]])
    assert cam_correlation(a, 2 * a) == pytest.approx(1.0)
    assert cam_correlation(np.ones(4), np.ones(4)) == 1.0
    assert cam_correlation(np.ones(4), a) == 0.0


def test_decide_rule() -> None:
    """int8 recommandé seulement si baisse <= 0,5 point ET corrélation > 0,95."""
    assert decide(0.3, 0.99) == "int8"
    assert decide(0.6, 0.99) == "fp32"
    assert decide(0.1, 0.90) == "fp32"
