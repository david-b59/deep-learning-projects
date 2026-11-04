"""Fixtures de test : un faux modèle exporté (poids aléatoires) et un client de test."""

from __future__ import annotations

import io
import json
from collections.abc import Iterator
from pathlib import Path

import pytest
import torch
from brain_mri.config import ModelConfig
from brain_mri.model import build_model
from brain_mri.predict import CONFIG_NAME, WEIGHTS_NAME
from fastapi.testclient import TestClient
from PIL import Image
from safetensors.torch import save_file

from app.config import Settings
from app.main import create_app

CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]


def make_model_dir(folder: Path, with_test_scores: bool = True) -> Path:
    """Écrit un dossier de modèle exporté (poids aléatoires, images 32x32) dans ``folder``."""
    torch.manual_seed(0)
    model = build_model(4, ModelConfig(pretrained=False))
    save_file({k: v.contiguous() for k, v in model.state_dict().items()}, folder / WEIGHTS_NAME)
    description = {
        "class_names": CLASSES,
        "preprocessing": {
            "image_size": 32,
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
    }
    if with_test_scores:
        scores = {"precision": 0.9, "recall": 0.8, "f1": 0.85, "support": 10}
        description["test"] = {
            "accuracy": 0.9,
            "per_class": {name: scores for name in CLASSES},
            "confusion_matrix": [[10 if i == j else 0 for j in range(4)] for i in range(4)],
        }
    (folder / CONFIG_NAME).write_text(json.dumps(description), encoding="utf-8")
    return folder


def image_bytes(fmt: str = "PNG", size: tuple[int, int] = (40, 50)) -> bytes:
    """Fabrique une petite image en niveaux de gris dans le format demandé."""
    buffer = io.BytesIO()
    Image.new("L", size, 128).save(buffer, format=fmt)
    return buffer.getvalue()


@pytest.fixture(scope="session")
def model_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Dossier du faux modèle, créé une seule fois pour toute la session de tests."""
    return make_model_dir(tmp_path_factory.mktemp("model"))


@pytest.fixture(scope="session")
def client(model_dir: Path) -> Iterator[TestClient]:
    """Client de test : l'application démarre (le modèle est chargé) une seule fois."""
    settings = Settings(
        model_dir=model_dir, model_repo_id="inutilise", cors_origin="http://front.test"
    )
    with TestClient(create_app(settings)) as test_client:  # `with` : exécute le démarrage
        yield test_client
