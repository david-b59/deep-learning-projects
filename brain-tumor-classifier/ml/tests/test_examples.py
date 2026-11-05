"""Tests de brain_mri.examples (tirage des images d'exemple de la démo)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from brain_mri.examples import CLASS_NAMES, MANIFEST_NAME, pick_examples


def test_pick_examples_copies_images_and_writes_manifest(raw_dir: Path, tmp_path: Path) -> None:
    """On obtient le bon nombre d'images par classe, copiées, et listées dans le manifeste."""
    out = tmp_path / "examples"
    manifest = pick_examples(raw_dir, out, per_class=3, seed=1)
    assert len(manifest) == 3 * len(CLASS_NAMES)
    for item in manifest:
        assert item["label"] in CLASS_NAMES
        assert (out / Path(item["src"]).name).is_file()
    assert json.loads((out / MANIFEST_NAME).read_text(encoding="utf-8")) == manifest


def test_pick_examples_is_reproducible_and_cleans_old_files(raw_dir: Path, tmp_path: Path) -> None:
    """Même graine = mêmes images ; un ancien fichier du dossier de sortie est supprimé."""
    out = tmp_path / "examples"
    out.mkdir()
    (out / "ancien.jpg").write_bytes(b"x")
    first = pick_examples(raw_dir, out, per_class=3, seed=7)
    second = pick_examples(raw_dir, out, per_class=3, seed=7)
    assert first == second
    assert not (out / "ancien.jpg").exists()


def test_pick_examples_rejects_too_many_per_class(raw_dir: Path, tmp_path: Path) -> None:
    """Demander plus d'images que la classe n'en contient donne une erreur claire."""
    with pytest.raises(ValueError, match="demandées"):
        pick_examples(raw_dir, tmp_path / "examples", per_class=1000)


def test_pick_examples_missing_folder_raises(tmp_path: Path) -> None:
    """Un jeu de test absent donne une erreur claire."""
    with pytest.raises(FileNotFoundError):
        pick_examples(tmp_path / "vide", tmp_path / "examples")
