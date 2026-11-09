"""Tests de ``brain_mri.audit`` sur de fausses images créées dans un dossier temporaire."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from brain_mri.audit import (
    accuracy_by_group,
    metadata_only_classifier,
    read_metadata,
    size_table,
)


def make_images(root: Path) -> None:
    """Crée deux classes : ``a`` en 512x512 RGB, ``b`` en 100x80 niveaux de gris."""
    for label, size, mode in (("a", (512, 512), "RGB"), ("b", (100, 80), "L")):
        folder = root / label
        folder.mkdir(parents=True)
        for k in range(4):
            Image.new(mode, size).save(folder / f"{k}.png")


def test_read_metadata(tmp_path: Path) -> None:
    """La taille, le mode et la classe de chaque image sont bien lus."""
    make_images(tmp_path)
    infos = read_metadata(tmp_path)
    assert len(infos) == 8
    by_label = {i.label: i for i in infos}
    assert (by_label["a"].width, by_label["a"].height, by_label["a"].mode) == (512, 512, "RGB")
    assert (by_label["b"].width, by_label["b"].height, by_label["b"].mode) == (100, 80, "L")


def test_size_table(tmp_path: Path) -> None:
    """Le tableau donne les bonnes parts de 512x512, de tailles distinctes et de RGB."""
    make_images(tmp_path)
    table = size_table(read_metadata(tmp_path))
    assert table["a"] == {"n": 4, "share_512": 1.0, "n_distinct_sizes": 1, "share_rgb": 1.0}
    assert table["b"]["share_512"] == 0.0
    assert table["b"]["share_rgb"] == 0.0


def test_accuracy_by_group(tmp_path: Path) -> None:
    """L'exactitude est séparée entre images 512x512 et autres ; un groupe vide vaut None."""
    make_images(tmp_path)
    infos = read_metadata(tmp_path)
    truth = [i.label for i in infos]
    pred = ["a"] * len(infos)  # le modèle répond toujours « a »
    result = accuracy_by_group(infos, truth, pred)
    assert result["a"]["512x512"]["accuracy"] == 1.0
    assert result["a"]["autre"]["accuracy"] is None
    assert result["b"]["autre"]["accuracy"] == 0.0


def test_metadata_only_classifier(tmp_path: Path) -> None:
    """Quand la taille sépare les classes, le classifieur idiot atteint 100 %."""
    make_images(tmp_path)
    infos = read_metadata(tmp_path)
    result = metadata_only_classifier(infos, infos)
    assert result["accuracy"] == 1.0
    assert result["chance_level"] == 0.5
    assert result["per_class"]["a"]["accuracy"] == 1.0
