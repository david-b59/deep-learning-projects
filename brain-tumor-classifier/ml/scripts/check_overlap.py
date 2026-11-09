"""Mesure le recouvrement d'un jeu externe avec Kaggle (Training + Testing), étape 11a.

Un jeu externe ne sert de test honnête que si ses images ne sont pas déjà dans le jeu
d'entraînement. Pour chaque image du jeu externe, le script :

    1. calcule (ou relit dans le cache ``data/cache/embeddings/``) son embedding DINOv2 de
       la tête recadrée, ainsi que ceux de tout Kaggle ;
    2. cherche son plus proche voisin Kaggle (toutes classes) par similarité cosinus ;
    3. la range dans une catégorie :
         - ``exact``     : même SHA-256 qu'une image Kaggle ;
         - ``copies``    : cosinus >= ``DUPLICATE_COSINE`` (0,94), hors exacts : copie recadrée,
                           zoomée ou recompressée de Kaggle, exclue par prudence ;
         - ``gray_zone`` : cosinus dans [``GRAY_ZONE_COSINE``, ``DUPLICATE_COSINE``[
                           (0,90 à 0,94) : à relire à l'œil sur une planche ;
         - ``kept``      : cosinus < 0,90, jugée absente de Kaggle ;
    4. repère les doublons INTERNES (deux images du jeu externe avec cosinus >= 0,94), utile
       pour séparer proprement un jeu de validation et un jeu de test.

Sorties :
    - ``docs/figures/overlap_<jeu>.json`` : décompte par classe, part retrouvée, classes des
      jumelles Kaggle, doublons internes ;
    - ``manifest.json`` du jeu mis à jour (``status``, ``cosine``, ``kaggle_match``,
      ``kaggle_class``, ``internal_twin``, ``internal_cosine``) ; AUCUN fichier n'est déplacé ;
    - trois planches dans ``data/cache/`` : ``overlap_<jeu>_gray_zone.png`` (16 paires de la
      zone grise), ``overlap_<jeu>_just_above.png`` (16 paires juste au-dessus de 0,94) et
      ``overlap_<jeu>_kept.png`` (24 images gardées tirées au hasard, graine 42, avec leur
      plus proche voisin Kaggle).

Utilisation (depuis ``brain-tumor-classifier/ml/``) ::

    uv run python scripts/check_overlap.py --dataset bdneuro
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
from download_external import CLASS_NAMES, SPECS
from PIL import Image, ImageDraw

from brain_mri.duplicates import sha256_file
from brain_mri.embeddings import MODEL_ID, MODEL_REVISION, embed_files, load_model
from brain_mri.head_crop import CROP_VERSION
from brain_mri.near_duplicates import (
    DUPLICATE_COSINE,
    GRAY_ZONE_COSINE,
    list_images,
    nearest_cosine,
    similar_pairs,
)

ROOT = Path(__file__).resolve().parents[2]  # dossier brain-tumor-classifier/
KAGGLE_DIR = ROOT / "data" / "raw"
CACHE_DIR = ROOT / "data" / "cache"
FIGURES_DIR = ROOT / "docs" / "figures"
SEED = 42  # graine du tirage des images gardées
PAIRS_PER_SHEET = 16  # paires par planche (zone grise, juste au-dessus du seuil)
KEPT_PER_CLASS = 6  # images gardées par classe sur la planche des gardées (24 au total)
JUST_ABOVE_POOL = 64  # on tire les 16 paires parmi les 64 images les plus proches du seuil


def kaggle_files() -> tuple[list[Path], list[str]]:
    """Liste toutes les images Kaggle (Training + Testing) et leur classe."""
    paths = [p for split in ("Training", "Testing") for p in list_images(KAGGLE_DIR / split)]
    return paths, [p.parent.name for p in paths]


def entry_path(root: Path, entry: dict) -> Path:
    """Chemin réel d'une image du manifeste (dans ``_excluded/`` si elle y a été déplacée)."""
    direct = root / entry["file"]
    return direct if direct.is_file() else root / "_excluded" / entry["file"]


def pair_sheet(
    path: Path, items: list[tuple[Path, Path, str, str]], title_height: int = 24
) -> None:
    """Dessine une planche de paires (image externe à gauche, voisin Kaggle à droite).

    Args:
        path: Fichier PNG à écrire.
        items: Liste de ``(image externe, image Kaggle, ligne 1, ligne 2)``.
        title_height: Hauteur de la bande de texte au-dessus de chaque paire.
    """
    tile, columns = 190, 4
    rows = (len(items) + columns - 1) // columns
    cell_w, cell_h = 2 * tile + 8, tile + title_height
    sheet = Image.new("RGB", (columns * cell_w, max(rows, 1) * cell_h), (30, 30, 30))
    draw = ImageDraw.Draw(sheet)
    for i, (left, right, line1, line2) in enumerate(items):
        r, c = divmod(i, columns)
        x, y = c * cell_w, r * cell_h
        for k, image_path in enumerate((left, right)):
            with Image.open(image_path) as image:
                sheet.paste(
                    image.convert("RGB").resize((tile, tile)), (x + k * tile, y + title_height)
                )
        draw.text((x + 2, y + 2), line1, fill=(255, 255, 0))
        draw.text((x + 2, y + 12), line2, fill=(200, 200, 200))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def summarize_twins(
    indices: list[int], twin_idx: np.ndarray, labels: list[str], kag_labels: list[str]
) -> dict:
    """Compte, pour un groupe d'images, la classe de leur jumelle Kaggle.

    Args:
        indices: Positions (dans le jeu externe) des images du groupe.
        twin_idx: Indice du plus proche voisin Kaggle de chaque image du jeu.
        labels: Classe de chaque image du jeu externe.
        kag_labels: Classe de chaque image Kaggle.

    Returns:
        ``{"same_class": n, "other_class": n, "pairs": {"classe jeu -> classe Kaggle": n}}``.
    """
    pairs = Counter(f"{labels[i]} -> {kag_labels[int(twin_idx[i])]}" for i in indices)
    same = sum(n for key, n in pairs.items() if key.split(" -> ")[0] == key.split(" -> ")[1])
    return {
        "same_class": same,
        "other_class": len(indices) - same,
        "pairs": dict(sorted(pairs.items())),
    }


def connected_groups(count: int, pairs: list[tuple[int, int, float]]) -> list[list[int]]:
    """Regroupe les images liées par des paires (composantes connexes, union-find).

    Args:
        count: Nombre d'images.
        pairs: Paires ``(i, j, cosinus)``.

    Returns:
        Les groupes de 2 images ou plus (listes d'indices triées).
    """
    parent = list(range(count))

    def find(a: int) -> int:
        """Racine du groupe de ``a`` (avec compression de chemin)."""
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i, j, _ in pairs:
        parent[find(i)] = find(j)
    groups: dict[int, list[int]] = {}
    for i in range(count):
        groups.setdefault(find(i), []).append(i)
    return [sorted(g) for g in groups.values() if len(g) > 1]


def main() -> None:
    """Calcule le recouvrement, écrit le rapport, met à jour le manifeste, dessine les planches."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dataset", choices=sorted(SPECS), required=True, help="jeu à contrôler")
    args = parser.parse_args()
    spec = SPECS[args.dataset]
    if spec.dedupe:
        raise SystemExit(f"{spec.key} est déjà dédoublonné au téléchargement (dHash) : non géré.")
    root = spec.directory
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = manifest["images"]
    paths = [entry_path(root, e) for e in entries]
    labels = [e["class"] for e in entries]

    kag_paths, kag_labels = kaggle_files()
    print(f"Kaggle : {len(kag_paths)} images ; {spec.key} : {len(paths)} images")

    cache = CACHE_DIR / "embeddings"
    last = [0.0]

    def progress(done: int, count: int) -> None:
        """Affiche l'avancement au plus toutes les 30 secondes."""
        if time.time() - last[0] > 30 or done == count:
            last[0] = time.time()
            print(f"  {done}/{count}", flush=True)

    model = load_model()
    print("Embeddings Kaggle (cache)...")
    kag_emb = embed_files(model, kag_paths, cache, progress=progress)
    print(f"Embeddings {spec.key}...")
    emb = embed_files(model, paths, cache, progress=progress)

    # 1. Doublons exacts (SHA-256) puis plus proche voisin Kaggle par cosinus (toutes classes)
    kag_sha = {sha256_file(p) for p in kag_paths}
    exact = np.array([e["sha256"] in kag_sha for e in entries])
    twin_idx, twin_cos = nearest_cosine(emb, kag_emb)

    status = np.where(
        exact,
        "excluded_exact",
        np.where(
            twin_cos >= DUPLICATE_COSINE,
            "excluded_copy",
            np.where(twin_cos >= GRAY_ZONE_COSINE, "gray_zone", "kept"),
        ),
    )

    # 2. Doublons internes au jeu externe
    internal = similar_pairs(emb, DUPLICATE_COSINE)
    internal_best: dict[int, tuple[int, float]] = {}
    for i, j, c in internal:
        for a, b in ((i, j), (j, i)):
            if a not in internal_best or c > internal_best[a][1]:
                internal_best[a] = (b, c)
    groups = connected_groups(len(entries), internal)
    cross_split = sum(
        1 for i, j, _ in internal if entries[i].get("split") != entries[j].get("split")
    )

    # 3. Rapport par classe
    per_class = {}
    for cls in CLASS_NAMES:
        mask = np.array([label == cls for label in labels])
        per_class[cls] = {
            "total": int(mask.sum()),
            "exact": int((mask & (status == "excluded_exact")).sum()),
            "copies": int((mask & (status == "excluded_copy")).sum()),
            "gray_zone": int((mask & (status == "gray_zone")).sum()),
            "kept": int((mask & (status == "kept")).sum()),
        }
    total = len(entries)
    found = int(((status == "excluded_exact") | (status == "excluded_copy")).sum())
    copy_idx = [
        int(i) for i in np.flatnonzero(np.isin(status, ["excluded_exact", "excluded_copy"]))
    ]
    gray_idx = [int(i) for i in np.flatnonzero(status == "gray_zone")]
    report = {
        "dataset": spec.title,
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "head_crop_version": CROP_VERSION,
        "thresholds": {"duplicate_cosine": DUPLICATE_COSINE, "gray_zone_cosine": GRAY_ZONE_COSINE},
        "per_class": per_class,
        "total": {
            "images": total,
            "exact": int((status == "excluded_exact").sum()),
            "copies": int((status == "excluded_copy").sum()),
            "gray_zone": len(gray_idx),
            "kept": int((status == "kept").sum()),
            "found_share": found / total,
            "found_or_gray_share": (found + len(gray_idx)) / total,
        },
        "twin_classes": {
            "copies_and_exact": summarize_twins(copy_idx, twin_idx, labels, kag_labels),
            "gray_zone": summarize_twins(gray_idx, twin_idx, labels, kag_labels),
        },
        "internal_duplicates": {
            "threshold": DUPLICATE_COSINE,
            "pairs": len(internal),
            "images_with_a_twin": len(internal_best),
            "images_with_a_twin_share": len(internal_best) / total,
            "groups": len(groups),
            "largest_group": max((len(g) for g in groups), default=0),
            "pairs_across_different_splits": cross_split,
            "images_with_a_twin_per_class": {
                cls: sum(labels[i] == cls for i in internal_best) for cls in CLASS_NAMES
            },
        },
    }
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    report_path = FIGURES_DIR / f"overlap_{spec.key}.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    # 4. Manifeste : statut, cosinus, jumelle Kaggle, jumelle interne (aucun fichier déplacé)
    for i, entry in enumerate(entries):
        entry["status"] = str(status[i])
        entry["cosine"] = round(float(twin_cos[i]), 4)
        entry["kaggle_match"] = kag_paths[int(twin_idx[i])].relative_to(KAGGLE_DIR).as_posix()
        entry["kaggle_class"] = kag_labels[int(twin_idx[i])]
        if i in internal_best:
            entry["internal_twin"] = entries[internal_best[i][0]]["file"]
            entry["internal_cosine"] = round(internal_best[i][1], 4)
        else:
            entry.pop("internal_twin", None)
            entry.pop("internal_cosine", None)
    manifest["overlap_check"] = {
        "duplicate_cosine": DUPLICATE_COSINE,
        "gray_zone_cosine": GRAY_ZONE_COSINE,
        "head_crop_version": CROP_VERSION,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    # 5. Planches
    def pair_item(i: int) -> tuple[Path, Path, str, str]:
        """Élément de planche pour l'image ``i`` : elle, son voisin Kaggle, deux lignes de texte."""
        j = int(twin_idx[i])
        line1 = f"cos {twin_cos[i]:.3f} {status[i]}"
        return paths[i], kag_paths[j], line1, f"{labels[i]} | Kaggle {kag_labels[j]}"

    def spread(indices: list[int], count: int) -> list[int]:
        """Choisit ``count`` éléments répartis régulièrement dans une liste triée par cosinus."""
        ordered = sorted(indices, key=lambda i: twin_cos[i])
        if len(ordered) <= count:
            return ordered
        return [ordered[k] for k in np.linspace(0, len(ordered) - 1, count).astype(int)]

    sheets = {
        "gray_zone": spread(gray_idx, PAIRS_PER_SHEET),
        "just_above": spread(
            sorted(
                (int(i) for i in np.flatnonzero(status == "excluded_copy")),
                key=lambda i: twin_cos[i],
            )[:JUST_ABOVE_POOL],
            PAIRS_PER_SHEET,
        ),
    }
    rng = random.Random(SEED)
    kept_pick: list[int] = []
    for cls in CLASS_NAMES:
        pool = [i for i in range(total) if status[i] == "kept" and labels[i] == cls]
        kept_pick += rng.sample(pool, min(KEPT_PER_CLASS, len(pool)))
    sheets["kept"] = kept_pick
    sheet_paths = []
    for name, indices in sheets.items():
        sheet_path = CACHE_DIR / f"overlap_{spec.key}_{name}.png"
        pair_sheet(sheet_path, [pair_item(i) for i in indices])
        sheet_paths.append(sheet_path)

    # 6. Résumé affiché
    print(f"\n{'classe':<11} {'total':>6} {'exacts':>7} {'copies':>7} {'grise':>6} {'gardées':>8}")
    for cls, c in per_class.items():
        print(
            f"{cls:<11} {c['total']:>6} {c['exact']:>7} {c['copies']:>7} "
            f"{c['gray_zone']:>6} {c['kept']:>8}"
        )
    print(f"\nPart retrouvée dans Kaggle (exacts + copies) : {found / total:.1%}")
    print(f"Doublons internes : {len(internal_best)} images, {len(groups)} groupes")
    print(f"Rapport : {report_path}\nManifeste : {manifest_path}")
    for sheet_path in sheet_paths:
        print(f"Planche : {sheet_path}")


if __name__ == "__main__":
    main()
