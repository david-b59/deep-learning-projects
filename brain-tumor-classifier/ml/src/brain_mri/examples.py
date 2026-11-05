"""Choix des images d'exemple de la démo du site : tirage au hasard dans le jeu de test.

Les images sont tirées **sans regarder si le modèle les classe bien** : montrer seulement des
réussites donnerait une image trop flatteuse. Le tirage est reproductible (graine fixe).

Lancer depuis ``brain-tumor-classifier/ml/`` ::

    uv run python -m brain_mri.examples
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

from brain_mri.config import PROJECT_ROOT, DataConfig

CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
MANIFEST_NAME = "manifest.json"


def pick_examples(
    raw_dir: Path,
    out_dir: Path,
    per_class: int = 6,
    seed: int = 42,
    manifest_path: Path | None = None,
) -> list[dict]:
    """Copie ``per_class`` images de test au hasard par classe et écrit la liste (manifeste).

    Args:
        raw_dir: Dossier contenant ``Testing/<classe>/`` (le jeu de test).
        out_dir: Dossier de sortie (``frontend/public/examples``), vidé d'abord.
        per_class: Nombre d'images par classe.
        seed: Graine du tirage (même graine = mêmes images).
        manifest_path: Où écrire la liste (manifeste) ; par défaut ``out_dir/manifest.json``.

    Returns:
        La liste des exemples : ``[{"label": classe, "src": "/examples/<fichier>"}, ...]``.

    Raises:
        FileNotFoundError: Si le dossier d'une classe est absent.
        ValueError: Si une classe contient moins d'images que demandé.
    """
    rng = random.Random(seed)

    # 1. Tirage et vérifications d'abord : en cas d'erreur, l'ancien dossier reste intact.
    chosen: list[tuple[str, Path]] = []
    for name in CLASS_NAMES:
        folder = raw_dir / "Testing" / name
        if not folder.is_dir():
            raise FileNotFoundError(f"Dossier de test introuvable : {folder}")
        files = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        if len(files) < per_class:
            raise ValueError(f"{name} : {len(files)} images, {per_class} demandées")
        chosen += [(name, source) for source in rng.sample(files, per_class)]

    # 2. Copie dans un dossier propre : pas d'anciens exemples qui traînent.
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    manifest: list[dict] = []
    counters = dict.fromkeys(CLASS_NAMES, 0)
    for name, source in chosen:
        counters[name] += 1
        target = f"{name}_{counters[name]:02d}{source.suffix.lower()}"
        shutil.copyfile(source, out_dir / target)
        manifest.append({"label": name, "src": f"/examples/{target}"})

    target_manifest = manifest_path or out_dir / MANIFEST_NAME
    target_manifest.parent.mkdir(parents=True, exist_ok=True)
    target_manifest.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    """Point d'entrée en ligne de commande."""
    parser = argparse.ArgumentParser(description="Tire les images d'exemple de la démo.")
    parser.add_argument("--per-class", type=int, default=6, help="images par classe")
    parser.add_argument("--seed", type=int, default=42, help="graine du tirage")
    args = parser.parse_args()
    out_dir = PROJECT_ROOT / "frontend" / "public" / "examples"
    # Un chemin relatif est résolu à partir de la racine du projet par DataConfig.
    raw_dir = DataConfig(raw_dir=Path("data/raw")).raw_dir
    # La liste est écrite dans le code du site (importée à la compilation) : pas de requête.
    manifest_path = PROJECT_ROOT / "frontend" / "src" / "lib" / "examples.json"
    manifest = pick_examples(raw_dir, out_dir, args.per_class, args.seed, manifest_path)
    print(f"{len(manifest)} images d'exemple copiées dans {out_dir}")


if __name__ == "__main__":
    main()
