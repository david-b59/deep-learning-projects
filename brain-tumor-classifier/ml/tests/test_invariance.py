"""Tests de brain_mri.invariance et du script evaluate_invariance (données synthétiques)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn

from brain_mri.config import DataConfig, ModelConfig
from brain_mri.invariance import (
    TRANSFORMATIONS,
    add_black_bands,
    analyze_pmram_by_twin_part,
    analyze_transformations,
    flip_summary,
    jpeg,
    measure_transformations,
    mirror,
    offcenter_crop,
    stretch_horizontal,
    zoom,
)
from brain_mri.model import build_model
from brain_mri.train import save_checkpoint
from brain_mri.transforms import build_transforms

CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]


class MeanModel(nn.Module):
    """Faux modèle : prédit la classe selon la luminosité moyenne du lot (4 classes)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Renvoie des scores dont l'argmax dépend de la luminosité moyenne de chaque image."""
        level = (x.mean(dim=(1, 2, 3)) * 10).long().clamp(0, 3)
        return torch.nn.functional.one_hot(level, 4).float()


def _image(width: int = 40, height: int = 30) -> Image.Image:
    """Image RGB aléatoire (graine fixe)."""
    rng = np.random.default_rng(0)
    return Image.fromarray(rng.integers(0, 256, (height, width, 3), dtype=np.uint8))


def test_fixed_transformations_geometry() -> None:
    """Chaque transformation fixe donne la bonne taille ; le miroir s'annule deux fois."""
    image = _image(40, 32)
    assert add_black_bands(image).size == (60, 48)  # +25 % de chaque côté
    assert zoom(image, 0.85).size == image.size and zoom(image, 1.15).size == image.size
    assert offcenter_crop(image).size == image.size
    assert stretch_horizontal(image, 1.2).size == (48, 32)
    assert stretch_horizontal(image, 0.8).size == (32, 32)
    assert jpeg(image).size == image.size and jpeg(image).mode == "RGB"
    assert np.array_equal(np.asarray(mirror(mirror(image))), np.asarray(image))
    assert not np.array_equal(np.asarray(mirror(image)), np.asarray(image))
    assert len(TRANSFORMATIONS) == 8


def test_zoom_out_adds_black_border() -> None:
    """Le zoom 0,85 laisse un cadre noir ; le coin de l'image est noir."""
    out = zoom(Image.new("RGB", (40, 40), (255, 255, 255)), 0.85)
    assert out.getpixel((0, 0)) == (0, 0, 0)
    assert out.getpixel((20, 20)) == (255, 255, 255)


def test_flip_summary() -> None:
    """Bascules, juste -> faux et faux -> juste sont comptés correctement."""
    labels = np.array([0, 0, 1, 1])
    base = np.array([0, 0, 0, 1])  # juste, juste, faux, juste
    new = np.array([0, 1, 1, 1])  # juste, faux, juste, juste
    res = flip_summary(labels, base, new)
    assert res == {
        "n": 4,
        "flips": 2,
        "flip_rate": 0.5,
        "right_to_wrong": 1,
        "wrong_to_right": 1,
    }
    assert flip_summary(labels[:0], base[:0], new[:0])["flip_rate"] is None


def test_analyze_transformations_per_class() -> None:
    """Le résultat est détaillé au total et par classe."""
    labels = np.array([0, 0, 1, 1])
    base = np.array([0, 0, 1, 1])
    res = analyze_transformations(labels, base, {"t": np.array([0, 1, 1, 1])}, ["a", "b"])
    block = res["transformations"]["t"]
    assert block["overall"]["flips"] == 1
    assert block["per_class"]["a"]["flip_rate"] == 0.5
    assert block["per_class"]["b"]["flip_rate"] == 0.0
    assert res["base_accuracy"] == 1.0


def test_measure_transformations_identity_never_flips(config: DataConfig, raw_dir: Path) -> None:
    """L'identité ne fait jamais basculer ; toutes les transformations sont mesurées."""
    paths = sorted((raw_dir / "Testing" / "glioma").glob("*.jpg"))
    labels = np.zeros(len(paths), dtype=np.int64)
    transform = build_transforms(config, train=False)
    res = measure_transformations(
        MeanModel(),
        paths,
        labels,
        transform,
        torch.device("cpu"),
        CLASS_NAMES,
        transformations={"identite": lambda image: image, **TRANSFORMATIONS},
    )
    assert res["transformations"]["identite"]["overall"]["flips"] == 0
    assert set(res["transformations"]) == {"identite", *TRANSFORMATIONS}
    assert res["transformations"]["miroir"]["overall"]["n"] == len(paths)


def test_analyze_pmram_by_twin_part() -> None:
    """Sans partie : un seul bloc « all » ; avec parties : un bloc par partie présente."""
    cos = np.array([0.99, 0.99, 0.99, 0.5])
    same = np.array([True, True, True, True])
    labels = np.array([0, 0, 1, 1])
    copy = np.array([0, 1, 1, 1])
    twin = np.array([0, 0, 1, 1])
    only = analyze_pmram_by_twin_part(cos, same, labels, copy, twin, None, ["a", "b"])
    assert list(only) == ["all"]
    assert only["all"]["safe"]["n"] == 3 and only["all"]["safe"]["flips"] == 1
    part = np.array(["train", "test", "test", "train"])
    split = analyze_pmram_by_twin_part(cos, same, labels, copy, twin, part, ["a", "b"])
    assert set(split) == {"all", "twin_in_train", "twin_in_test"}
    assert split["twin_in_train"]["safe"]["n"] == 1  # la 4e image a un cosinus trop bas
    assert split["twin_in_test"]["safe"]["n"] == 2
    assert split["twin_in_test"]["safe"]["flips"] == 1


def _load_script():
    """Charge scripts/evaluate_invariance.py comme un module (scripts n'est pas un paquet)."""
    path = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_invariance.py"
    spec = importlib.util.spec_from_file_location("evaluate_invariance", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_script_runs_on_synthetic_data(raw_dir: Path, tmp_path: Path) -> None:
    """Le script mesure le Testing synthétique avec un ResNet-50 aléatoire et écrit son JSON."""
    model = build_model(4, ModelConfig(pretrained=False, freeze_backbone=False))
    checkpoint = tmp_path / "model.pt"
    save_checkpoint(checkpoint, model, CLASS_NAMES, {})
    config_file = tmp_path / "train.yaml"
    config_file.write_text(
        f"data:\n  raw_dir: {raw_dir.as_posix()}\n  image_size: 64\n  batch_size: 8\n",
        encoding="utf-8",
    )
    out = tmp_path / "invariance.json"
    script = _load_script()
    report = script.main(
        [
            "--config", str(config_file),
            "--checkpoint", str(checkpoint),
            "--preprocessing", "head_crop",
            "--skip-pmram",
            "--max-images", "6",
            "--output", str(out),
        ]
    )  # fmt: skip
    saved = json.loads(out.read_text(encoding="utf-8"))
    assert saved["preprocessing"] == "head_crop"
    assert set(saved["transformations"]["transformations"]) == set(TRANSFORMATIONS)
    assert report["transformations"]["n"] == 6
