"""Calibre les seuils de détection des quasi-doublons (embeddings DINOv2 et dHash 256 bits).

Question : à partir de quelle ressemblance deux images sont-elles la MÊME image ? On le
mesure sur des cas où la réponse est connue :

    - positifs : les paires PMRAM <-> Kaggle que le dHash 64 bits (distance <= 2) a déjà
      reconnues comme identiques (liste dans ``data/external/pmram/manifest.json``) ;
    - négatifs : pour chaque image PMRAM (et chaque image Kaggle), son plus proche voisin
      Kaggle de CLASSE DIFFÉRENTE. Une même image ne peut pas avoir deux classes : si une
      telle paire est jugée « doublon », c'est une erreur du détecteur.

Le script :
    1. mesure la vitesse sur 64 images et affiche la durée estimée ;
    2. calcule (ou relit dans le cache ``data/cache/embeddings/``) les embeddings de tout
       Kaggle (Training + Testing) et de tout PMRAM brut, ainsi que leurs dHash 256 bits ;
    3. calcule deux seuils par méthode : « doublon sûr » (>= 99 % des positifs retrouvés
       et <= 0,1 % des négatifs) et « zone grise » ;
    4. mesure aussi, aux seuils retenus (``DUPLICATE_COSINE``, ``GRAY_ZONE_COSINE``), la part de
       positifs retrouvés ;
    5. écrit ``docs/figures/near_duplicate_calibration.json`` (seuils, taux, histogrammes)
       et la planche ``data/cache/gray_zone_pmram.png`` (16 paires de la zone grise).

Utilisation (depuis ``brain-tumor-classifier/ml/``) ::

    uv run python scripts/calibrate_near_duplicates.py [--speed-only]
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from brain_mri.embeddings import MODEL_ID, MODEL_REVISION, embed_files, load_model
from brain_mri.near_duplicates import (
    DUPLICATE_COSINE,
    GRAY_ZONE_COSINE,
    dhash256,
    hamming_nearest,
    list_images,
    nearest_cosine,
)

ROOT = Path(__file__).resolve().parents[2]  # dossier brain-tumor-classifier/
KAGGLE_DIR = ROOT / "data" / "raw"
PMRAM_DIR = ROOT / "data" / "external" / "pmram"
CACHE_DIR = ROOT / "data" / "cache"
REPORT_PATH = ROOT / "docs" / "figures" / "near_duplicate_calibration.json"
SHEET_PATH = CACHE_DIR / "gray_zone_pmram.png"

MIN_RECALL = 0.99  # « doublon sûr » : au moins 99 % des positifs retrouvés...
MAX_NEGATIVE_RATE = 0.001  # ... et au plus 0,1 % des négatifs
GRAY_NEGATIVE_RATE = 0.02  # « zone grise » : jusqu'à 2 % des négatifs acceptés pour ne rien rater


def kaggle_files() -> tuple[list[Path], np.ndarray]:
    """Liste toutes les images Kaggle (Training + Testing) et leur classe."""
    paths = [p for split in ("Training", "Testing") for p in list_images(KAGGLE_DIR / split)]
    return paths, np.array([p.parent.name for p in paths])


def pmram_files() -> tuple[list[Path], np.ndarray, dict]:
    """Liste toutes les images PMRAM brutes (gardées ET écartées), leur classe et le manifeste.

    Returns:
        ``(chemins, classes, manifeste)``. Les images écartées sont dans ``_excluded/``.
    """
    manifest = json.loads((PMRAM_DIR / "manifest.json").read_text(encoding="utf-8"))
    paths, labels = [], []
    for item in manifest["images"]:
        folder = PMRAM_DIR if item["status"] == "kept" else PMRAM_DIR / "_excluded"
        paths.append(folder / item["file"])
        labels.append(item["class"])
    return paths, np.array(labels), manifest


def dhash_array(paths: list[Path]) -> np.ndarray:
    """Calcule le dHash 256 bits (32 octets) de chaque image : tableau ``(N, 32)``."""
    out = []
    for path in paths:
        with Image.open(path) as image:
            out.append(dhash256(image))
    return np.stack(out)


def measure_speed(model, paths: list[Path]) -> float:
    """Mesure le temps par image (secondes) sur 64 images, avec un cache jetable."""
    probe = CACHE_DIR / "speedtest"
    shutil.rmtree(probe, ignore_errors=True)
    start = time.perf_counter()
    embed_files(model, paths[:64], probe)
    per_image = (time.perf_counter() - start) / 64
    shutil.rmtree(probe, ignore_errors=True)
    return per_image


def rate_curve(values: np.ndarray, thresholds: np.ndarray, higher_is_closer: bool) -> np.ndarray:
    """Part des valeurs jugées « doublon » pour chaque seuil (cosinus >= t ou distance <= t)."""
    if higher_is_closer:
        return np.array([(values >= t).mean() for t in thresholds])
    return np.array([(values <= t).mean() for t in thresholds])


def choose_thresholds(positives: np.ndarray, negatives: np.ndarray, higher_is_closer: bool) -> dict:
    """Choisit les seuils « doublon sûr » et « zone grise » pour une mesure de ressemblance.

    - doublon sûr : seuil le plus permissif qui garde au plus ``MAX_NEGATIVE_RATE`` de
      négatifs ; on y lit la part de positifs retrouvés (objectif : ``MIN_RECALL``) ;
    - zone grise : seuil le plus permissif qui garde au plus ``GRAY_NEGATIVE_RATE`` de négatifs.

    Args:
        positives: Mesures sur les paires de vrais doublons.
        negatives: Mesures sur les paires de classes différentes.
        higher_is_closer: Vrai pour le cosinus (grand = proche), faux pour une distance.

    Returns:
        Dictionnaire de seuils et de taux.
    """
    if higher_is_closer:
        candidates = np.unique(np.round(np.concatenate([positives, negatives]), 4))[::-1]
    else:
        candidates = np.unique(np.concatenate([positives, negatives]))

    def loosest(max_rate: float) -> float:
        """Seuil le plus permissif dont le taux de négatifs reste sous ``max_rate``."""
        best = candidates[0]  # on part du plus strict
        for t in candidates:
            if rate_curve(negatives, np.array([t]), higher_is_closer)[0] <= max_rate:
                best = t
            else:
                break
        return float(best)

    sure, gray = loosest(MAX_NEGATIVE_RATE), loosest(GRAY_NEGATIVE_RATE)

    def rates(t: float) -> dict:
        """Taux de positifs retrouvés et de négatifs retenus au seuil ``t``."""
        pair = np.array([t])
        return {
            "threshold": t,
            "positives_found": float(rate_curve(positives, pair, higher_is_closer)[0]),
            "negatives_kept": float(rate_curve(negatives, pair, higher_is_closer)[0]),
        }

    return {
        "direction": "cosinus >= seuil" if higher_is_closer else "distance <= seuil",
        "sure_duplicate": rates(sure),
        "gray_zone_limit": rates(gray),
        "sure_goal_met": rates(sure)["positives_found"] >= MIN_RECALL,
    }


def histogram(values: np.ndarray, edges: np.ndarray) -> dict:
    """Histogramme en chiffres : bornes des cases et effectif de chacune."""
    counts, _ = np.histogram(values, bins=edges)
    return {"edges": [round(float(e), 4) for e in edges], "counts": counts.tolist()}


def gray_zone_sheet(
    pairs: list[tuple[Path, Path, float, int]], labels: list[tuple[str, str]]
) -> None:
    """Dessine la planche des paires de la zone grise (PMRAM à gauche, Kaggle à droite)."""
    tile, columns = 170, 4
    rows = (len(pairs) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * (2 * tile + 8), rows * (tile + 24)), (30, 30, 30))
    draw = ImageDraw.Draw(sheet)
    for i, (left, right, sim, dist) in enumerate(pairs):
        r, c = divmod(i, columns)
        x, y = c * (2 * tile + 8), r * (tile + 24)
        for k, path in enumerate((left, right)):
            with Image.open(path) as image:
                sheet.paste(image.convert("RGB").resize((tile, tile)), (x + k * tile, y + 24))
        draw.text((x + 2, y + 2), f"cos {sim:.3f} dH {dist}", fill=(255, 255, 0))
        draw.text((x + 2, y + 12), f"{labels[i][0]} | {labels[i][1]}", fill=(200, 200, 200))
    SHEET_PATH.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(SHEET_PATH)


def main() -> None:
    """Calcule les embeddings et les dHash, calibre les seuils, écrit le rapport et la planche."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--speed-only", action="store_true", help="mesure la vitesse puis s'arrête")
    args = parser.parse_args()

    kag_paths, kag_labels = kaggle_files()
    pm_paths, pm_labels, manifest = pmram_files()
    print(f"Kaggle : {len(kag_paths)} images ; PMRAM brut : {len(pm_paths)} images")

    model = load_model()
    per_image = measure_speed(model, kag_paths)
    total = len(kag_paths) + len(pm_paths)
    print(
        f"Vitesse : {per_image * 1000:.0f} ms/image -> estimation {per_image * total / 60:.1f} min"
    )
    if args.speed_only:
        return

    cache = CACHE_DIR / "embeddings"
    last = [0.0]

    def progress(done: int, count: int) -> None:
        """Affiche l'avancement au plus toutes les 30 secondes."""
        if time.time() - last[0] > 30 or done == count:
            last[0] = time.time()
            print(f"  {done}/{count}", flush=True)

    print("Embeddings Kaggle...")
    kag_emb = embed_files(model, kag_paths, cache, progress=progress)
    print("Embeddings PMRAM...")
    pm_emb = embed_files(model, pm_paths, cache, progress=progress)
    print("dHash 256 bits...")
    kag_hash, pm_hash = dhash_array(kag_paths), dhash_array(pm_paths)

    # Positifs : paires PMRAM <-> Kaggle reconnues par le dHash 64 bits (distance <= 2)
    kag_index = {p.relative_to(KAGGLE_DIR).as_posix(): i for i, p in enumerate(kag_paths)}
    pos_pm, pos_kag = [], []
    for i, item in enumerate(manifest["images"]):
        if item["status"] != "kept" and item["distance"] <= 2:
            pos_pm.append(i)
            pos_kag.append(kag_index[item["kaggle_match"]])
    pos_cos = np.einsum("ij,ij->i", pm_emb[pos_pm], kag_emb[pos_kag])
    pos_ham = np.bitwise_count(pm_hash[pos_pm] ^ kag_hash[pos_kag]).sum(axis=1)
    print(f"Positifs : {len(pos_pm)} paires")

    # Négatifs : plus proche voisin Kaggle de classe différente
    _, neg_cos_pm = nearest_cosine(pm_emb, kag_emb, pm_labels, kag_labels, other_class=True)
    _, neg_cos_kag = nearest_cosine(kag_emb, kag_emb, kag_labels, kag_labels, other_class=True)
    neg_ham_pm = np.zeros(len(pm_paths), dtype=np.int64)
    neg_ham_kag = np.zeros(len(kag_paths), dtype=np.int64)
    for cls in np.unique(kag_labels):
        others = kag_hash[kag_labels != cls]
        for hashes, labels, out in (
            (pm_hash, pm_labels, neg_ham_pm),
            (kag_hash, kag_labels, neg_ham_kag),
        ):
            mask = labels == cls
            out[mask] = hamming_nearest(hashes[mask], others)[1]
    neg_cos = np.concatenate([neg_cos_pm, neg_cos_kag])
    neg_ham = np.concatenate([neg_ham_pm, neg_ham_kag])

    cosine = choose_thresholds(pos_cos, neg_cos, higher_is_closer=True)
    dhash = choose_thresholds(pos_ham.astype(float), neg_ham.astype(float), higher_is_closer=False)

    report = {
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "counts": {
            "kaggle_images": len(kag_paths),
            "pmram_images": len(pm_paths),
            "positive_pairs": len(pos_pm),
            "negative_pairs_pmram_vs_kaggle": len(neg_cos_pm),
            "negative_pairs_kaggle_vs_kaggle": len(neg_cos_kag),
        },
        "goal": {
            "min_recall": MIN_RECALL,
            "max_negative_rate": MAX_NEGATIVE_RATE,
            "gray_negative_rate": GRAY_NEGATIVE_RATE,
        },
        "cosine": cosine,
        "chosen_cosine_thresholds": {
            "duplicate": {
                "threshold": DUPLICATE_COSINE,
                "positives_found": float((pos_cos >= DUPLICATE_COSINE).mean()),
                "negatives_kept": float((neg_cos >= DUPLICATE_COSINE).mean()),
            },
            "gray_zone_limit": {
                "threshold": GRAY_ZONE_COSINE,
                "positives_found": float((pos_cos >= GRAY_ZONE_COSINE).mean()),
                "negatives_kept": float((neg_cos >= GRAY_ZONE_COSINE).mean()),
            },
        },
        "dhash256": dhash,
        "histograms": {
            "cosine_positives": histogram(pos_cos, np.linspace(0.5, 1.0, 51)),
            "cosine_negatives": histogram(neg_cos, np.linspace(0.5, 1.0, 51)),
            "dhash256_positives": histogram(pos_ham, np.arange(0, 132, 4)),
            "dhash256_negatives": histogram(neg_ham, np.arange(0, 132, 4)),
        },
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("cosine", "chosen_cosine_thresholds")}, indent=2))

    # Planche : paires PMRAM <-> meilleur voisin Kaggle (toutes classes) dans la zone grise
    best_idx, best_cos = nearest_cosine(pm_emb, kag_emb)
    low, high = GRAY_ZONE_COSINE, DUPLICATE_COSINE  # seuils retenus (constantes documentées)
    gray = np.flatnonzero((best_cos >= low) & (best_cos < high))
    print(f"Zone grise cosinus [{low:.4f}, {high:.4f}[ : {len(gray)} images PMRAM")
    if len(gray):
        chosen = gray[np.argsort(best_cos[gray])][np.linspace(0, len(gray) - 1, 16).astype(int)]
        pairs = []
        labels = []
        for i in chosen:
            j = int(best_idx[i])
            dist = int(np.bitwise_count(pm_hash[i] ^ kag_hash[j]).sum())
            pairs.append((pm_paths[i], kag_paths[j], float(best_cos[i]), dist))
            labels.append((f"PMRAM {pm_labels[i]}", f"Kaggle {kag_labels[j]}"))
        gray_zone_sheet(pairs, labels)
        print(f"Planche : {SHEET_PATH}")


if __name__ == "__main__":
    main()
