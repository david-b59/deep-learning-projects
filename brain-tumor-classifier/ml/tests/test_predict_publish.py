"""Tests de brain_mri.predict et brain_mri.publish (poids aléatoires, aucun envoi réseau)."""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import matplotlib
import numpy as np
import pytest
import torch
from PIL import Image
from safetensors.torch import load_file

from brain_mri.config import Config, DataConfig, ModelConfig, TrainConfig
from brain_mri.evaluate import evaluate
from brain_mri.model import build_model
from brain_mri.predict import (
    CONFIG_NAME,
    DISCLAIMER,
    WEIGHTS_NAME,
    Predictor,
    encode_png_base64,
    jet_colors,
    overlay_heatmap,
)
from brain_mri.publish import export_model, push_to_hub, resolve_repo_id
from brain_mri.train import save_checkpoint

CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]


@pytest.fixture
def full_config(config: DataConfig, tmp_path: Path) -> Config:
    """Configuration complète pour le faux jeu : sorties dans tmp_path."""
    training = TrainConfig(
        checkpoint_path=tmp_path / "models" / "best.pt",
        figures_dir=tmp_path / "figures",
        mlflow_dir=tmp_path / "mlruns",
    )
    return Config(data=config, model=ModelConfig(pretrained=False), training=training)


@pytest.fixture
def checkpoint(full_config: Config) -> Path:
    """Enregistre un « meilleur modèle » à poids aléatoires, avec son prétraitement."""
    torch.manual_seed(0)
    model = build_model(4, ModelConfig(pretrained=False))
    info = {
        "image_size": 32,
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "phase": "phase2",
        "epoch": 4,
        "val_loss": 0.5,
        "val_acc": 0.9,
    }
    save_checkpoint(full_config.training.checkpoint_path, model, CLASS_NAMES, info)
    return full_config.training.checkpoint_path


@pytest.fixture
def model_card(tmp_path: Path) -> Path:
    """Crée une fausse fiche de modèle."""
    path = tmp_path / "model_card.md"
    path.write_text("# Fiche de test\n", encoding="utf-8")
    return path


def _mri(mode: str = "L") -> Image.Image:
    """Fabrique une petite fausse IRM aléatoire dans le mode de couleur demandé."""
    rng = np.random.default_rng(1)
    return Image.fromarray(rng.integers(0, 256, (40, 50), dtype=np.uint8)).convert(mode)


# ----- predict -------------------------------------------------------------


def test_predict_returns_valid_probabilities_and_overlay(checkpoint: Path) -> None:
    """La prédiction donne 4 probabilités qui somment à 1, la classe la plus probable et un PNG."""
    predictor = Predictor.from_checkpoint(checkpoint, torch.device("cpu"))
    result = predictor.predict(_mri())
    assert set(result.probabilities) == set(CLASS_NAMES)
    assert sum(result.probabilities.values()) == pytest.approx(1.0, abs=1e-5)
    assert result.predicted_class == max(result.probabilities, key=result.probabilities.get)
    assert result.disclaimer == DISCLAIMER
    png = Image.open(io.BytesIO(base64.b64decode(result.gradcam_png_base64)))
    assert png.size == (32, 32) and png.mode == "RGB"  # taille imposée par le prétraitement


@pytest.mark.parametrize("mode", ["L", "RGB", "RGBA"])
def test_predict_accepts_any_color_mode(checkpoint: Path, mode: str) -> None:
    """Niveaux de gris, RGB et RGBA sont acceptés (convertis en RGB)."""
    predictor = Predictor.from_checkpoint(checkpoint, torch.device("cpu"))
    assert predictor.predict(_mri(mode)).predicted_class in CLASS_NAMES


def test_predict_is_deterministic_and_model_stays_in_eval(checkpoint: Path) -> None:
    """Deux prédictions de la même image sont identiques et le modèle reste en eval()."""
    predictor = Predictor.from_checkpoint(checkpoint, torch.device("cpu"))
    first, second = predictor.predict(_mri()), predictor.predict(_mri())
    assert first.probabilities == second.probabilities
    assert predictor.model.training is False


def test_overlay_heatmap_blends_image_and_map() -> None:
    """La superposition garde la taille de l'image et change ses couleurs."""
    image = np.zeros((8, 8, 3), dtype=np.float32)
    overlay = overlay_heatmap(image, np.ones((8, 8), dtype=np.float32))
    assert overlay.size == (8, 8)
    assert np.asarray(overlay).max() > 0  # la carte colore l'image noire


def test_encode_png_base64_roundtrip() -> None:
    """L'image encodée puis décodée a la même taille."""
    image = Image.new("RGB", (6, 4), "red")
    decoded = Image.open(io.BytesIO(base64.b64decode(encode_png_base64(image))))
    assert decoded.size == (6, 4)


# ----- export --------------------------------------------------------------


def test_export_writes_safetensors_config_and_card(
    full_config: Config, checkpoint: Path, model_card: Path, tmp_path: Path
) -> None:
    """L'export produit les poids, la configuration et la fiche du modèle."""
    out = export_model(full_config, tmp_path / "export", model_card)
    assert {p.name for p in out.iterdir()} == {WEIGHTS_NAME, CONFIG_NAME, "README.md"}
    description = json.loads((out / CONFIG_NAME).read_text(encoding="utf-8"))
    assert description["class_names"] == CLASS_NAMES
    assert description["preprocessing"]["image_size"] == 32
    assert description["selected_on_validation"]["val_acc"] == 0.9
    assert "test" not in description  # pas d'evaluation.json : pas de scores de test
    assert (out / "README.md").read_text(encoding="utf-8").startswith("# Fiche")


def test_export_includes_test_scores_when_available(
    full_config: Config, checkpoint: Path, model_card: Path, tmp_path: Path
) -> None:
    """Si l'évaluation a été faite, ses scores sont recopiés dans config.json."""
    evaluate(full_config)
    out = export_model(full_config, tmp_path / "export", model_card)
    description = json.loads((out / CONFIG_NAME).read_text(encoding="utf-8"))
    assert 0.0 <= description["test"]["accuracy"] <= 1.0
    assert set(description["test"]["per_class"]) == set(CLASS_NAMES)


def test_exported_folder_gives_the_same_predictions(
    full_config: Config, checkpoint: Path, model_card: Path, tmp_path: Path
) -> None:
    """Recharger le dossier exporté donne exactement les mêmes poids et les mêmes probabilités."""
    out = export_model(full_config, tmp_path / "export", model_card)
    original = Predictor.from_checkpoint(checkpoint, torch.device("cpu"))
    reloaded = Predictor.from_folder(out, torch.device("cpu"))
    weights = load_file(out / WEIGHTS_NAME)
    assert torch.equal(weights["fc.weight"], original.model.fc.weight.detach())
    assert original.predict(_mri()).probabilities == reloaded.predict(_mri()).probabilities


def test_export_without_checkpoint_or_card_raises(
    full_config: Config, model_card: Path, tmp_path: Path
) -> None:
    """Sans modèle entraîné, ou sans fiche, l'erreur est claire."""
    with pytest.raises(FileNotFoundError):
        export_model(full_config, tmp_path / "export", model_card)
    with pytest.raises(FileNotFoundError):
        export_model(full_config, tmp_path / "export", tmp_path / "absente.md")


def test_from_folder_missing_file_raises(tmp_path: Path) -> None:
    """Un dossier incomplet est refusé avec le nom du fichier manquant."""
    with pytest.raises(FileNotFoundError, match="model.safetensors"):
        Predictor.from_folder(tmp_path)


# ----- publication (sans réseau) ------------------------------------------


def test_push_without_token_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Sans HF_TOKEN, la publication s'arrête avant tout envoi."""
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr("brain_mri.publish.load_dotenv", lambda *a, **k: False)
    with pytest.raises(RuntimeError, match="HF_TOKEN"):
        push_to_hub(tmp_path)


def test_resolve_repo_id_uses_env_or_account_name(monkeypatch: pytest.MonkeyPatch) -> None:
    """L'identifiant du dépôt vient de HF_REPO_ID, sinon du nom du compte."""

    class FakeApi:
        """Faux client Hugging Face : renvoie un nom de compte fixe."""

        def whoami(self) -> dict:
            """Renvoie un compte de test."""
            return {"name": "alice"}

    monkeypatch.delenv("HF_REPO_ID", raising=False)
    assert resolve_repo_id(FakeApi()) == "alice/brain-tumor-classifier"
    monkeypatch.setenv("HF_REPO_ID", "autre/depot")
    assert resolve_repo_id(FakeApi()) == "autre/depot"


def test_jet_colors_matches_matplotlib() -> None:
    """La palette « jet » en NumPy reste à 1 niveau de gris près de celle de matplotlib."""
    values = np.arange(256) / 255  # 256 valeurs régulières entre 0 et 1
    ours = jet_colors(values) * 255
    reference = matplotlib.colormaps["jet"](values)[..., :3] * 255
    assert np.abs(ours - reference).max() <= 1
