"""Planche des tumeurs et des images saines hors 512 x 512 du test, à marquer à l'œil (11b-6a).

Montre les tumeurs hors 512 du test (toutes) et 60 « notumor » hors 512 tirées au hasard
(graine 42). Chaque vignette porte : numéro, vraie classe, prédiction de A0, taille d'origine.
Les pages PNG sont écrites dans ``data/cache/source_sheet_<k>.png`` (hors git). Un CSV modèle
``ml/splits/test_sequence_tags.csv`` (colonnes ``file, class, a0_pred, sequence``) liste les
images dans l'ordre des numéros (ligne k = vignette k) ; la colonne ``sequence`` est vide :
à remplir à l'œil (T1c / T1 / T2 / FLAIR / incertain).

Utilisation (depuis ``brain-tumor-classifier/ml/``) ::

    uv run python scripts/make_source_sheet.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]  # dossier brain-tumor-classifier/
MANIFEST = ROOT / "ml" / "splits" / "kaggle_grouped.csv"
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data" / "cache"
CSV_OUT = ROOT / "ml" / "splits" / "test_sequence_tags.csv"
SEED = 42
N_NOTUMOR = 60
THUMB = 160  # côté d'une vignette en pixels
COLUMNS, ROWS = 6, 4  # vignettes par page : 24
CAPTION = 44  # hauteur du texte sous chaque vignette


def a0_predictions() -> dict[str, str]:
    """Prédictions de A0 sur le test, reconstruites depuis ``evaluation.json``.

    Returns:
        ``{fichier: classe prédite}`` pour les seules erreurs ; les autres sont justes.
    """
    report = json.loads((ROOT / "docs/figures/a0/evaluation.json").read_text(encoding="utf-8"))
    return {e["file"]: e["predicted"] for e in report["errors"]}


def main() -> None:
    """Choisit les images, écrit le CSV modèle et les pages de la planche."""
    with open(MANIFEST, encoding="utf-8", newline="") as file:
        rows = [r for r in csv.DictReader(file) if r["part"] == "test" and r["is_512"] == "0"]
    tumors = [r for r in rows if r["class"] != "notumor"]
    healthy = [r for r in rows if r["class"] == "notumor"]
    picked = np.random.default_rng(SEED).choice(len(healthy), size=N_NOTUMOR, replace=False)
    chosen = tumors + [healthy[i] for i in sorted(picked)]
    errors = a0_predictions()
    items = [(r["file"], r["class"], errors.get(r["file"], r["class"])) for r in chosen]
    print(f"{len(tumors)} tumeurs hors 512 + {N_NOTUMOR} notumor hors 512 = {len(items)} images")

    with open(CSV_OUT, "w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["file", "class", "a0_pred", "sequence"])  # sequence : à remplir
        writer.writerows([(f, c, p, "") for f, c, p in items])

    font = ImageFont.load_default(size=13)
    per_page = COLUMNS * ROWS
    cell_h = THUMB + CAPTION
    for page, start in enumerate(range(0, len(items), per_page), start=1):
        sheet = Image.new("RGB", (COLUMNS * THUMB, ROWS * cell_h), (30, 30, 30))
        draw = ImageDraw.Draw(sheet)
        for k, (name, true, pred) in enumerate(items[start : start + per_page]):
            number = start + k + 1
            x, y = (k % COLUMNS) * THUMB, (k // COLUMNS) * cell_h
            with Image.open(RAW_DIR / name) as image:
                size = image.size
                thumb = image.convert("L").resize((THUMB, THUMB))  # étiré : repérage seulement
            sheet.paste(thumb.convert("RGB"), (x, y))
            color = (120, 255, 120) if pred == true else (255, 120, 120)
            draw.text((x + 3, y + THUMB + 2), f"#{number} {true}", fill=(255, 255, 255), font=font)
            draw.text((x + 3, y + THUMB + 15), f"A0 : {pred}", fill=color, font=font)
            draw.text(
                (x + 3, y + THUMB + 28), f"{size[0]}x{size[1]}", fill=(200, 200, 200), font=font
            )
        path = OUT_DIR / f"source_sheet_{page}.png"
        sheet.save(path)
        print(f"Page {page} : {path.relative_to(ROOT).as_posix()}")
    print(f"CSV modèle : {CSV_OUT.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
