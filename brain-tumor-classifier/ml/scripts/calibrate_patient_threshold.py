"""Cale le seuil de cosinus « même patient » sur de vrais identifiants (étape 11b-1).

Les images Kaggle n'ont pas d'identifiant de patient, mais une partie d'entre elles sont des
copies d'images figshare (cosinus DINOv2 >= 0,98, même classe) qui, elles, en ont un. Parmi les
paires Kaggle dont les DEUX images ont un identifiant connu, on mesure, pour chaque seuil de
cosinus, quelle part des paires est réellement du même patient.

Règle de décision, écrite AVANT de mesurer et appliquée mécaniquement :
    seuil retenu = la PLUS BASSE des tranches (0,90 / 0,92 / 0,93 / 0,94 / 0,95 / 0,96 / 0,98)
    où (a) au moins 95 % des paires de même classe, d'identifiants connus, sont du même
    patient ET (b) le plus grand groupe (union-find avec les liens patient) fait au plus 5 %
    des images ; si aucune tranche ne convient : 0,94.

Sortie : ``docs/figures/patient_threshold_calibration.json``.

Utilisation (depuis ``brain-tumor-classifier/ml/``) ::

    uv run python scripts/calibrate_patient_threshold.py
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np

from brain_mri.duplicates import sha256_file
from brain_mri.embeddings import embed_files, load_model
from brain_mri.grouping import LINK_COSINE, build_groups, match_patient_ids
from brain_mri.near_duplicates import list_images, similar_pairs

ROOT = Path(__file__).resolve().parents[2]  # dossier brain-tumor-classifier/
RAW_DIR = ROOT / "data" / "raw"
CACHE_DIR = ROOT / "data" / "cache"
FIGSHARE_DIR = ROOT / "data" / "external" / "figshare_pid"
FIGURES_DIR = ROOT / "docs" / "figures"
CLASS_NAMES = ("glioma", "meningioma", "notumor", "pituitary")  # même ordre que le modèle
TRANCHES = (0.90, 0.92, 0.93, 0.94, 0.95, 0.96, 0.98)  # seuils testés
MIN_SAME_PATIENT = 0.95  # règle (a) : part minimale de paires du même patient
MAX_GROUP_SHARE = 0.05  # règle (b) : part maximale du plus grand groupe
FALLBACK = 0.94  # seuil si aucune tranche ne satisfait la règle


def progress(done: int, count: int) -> None:
    """Affiche l'avancement des embeddings seulement à la fin (le cache est normalement plein)."""
    if done == count:
        print(f"  embeddings {done}/{count}", flush=True)


def load_inputs() -> dict:
    """Charge Kaggle (Training + Testing) et figshare avec leurs embeddings.

    Returns:
        Dictionnaire : ``paths`` (chemins Kaggle), ``rel`` (chemins relatifs à ``data/raw``),
        ``labels`` (indices de classe), ``sha`` (SHA-256), ``emb``, ``fig_emb``,
        ``fig_labels`` et ``fig_patients`` (figshare).
    """
    paths = [p for split in ("Training", "Testing") for p in list_images(RAW_DIR / split)]
    index = {name: i for i, name in enumerate(CLASS_NAMES)}
    labels = np.array([index[p.parent.name] for p in paths])
    rel = [p.relative_to(RAW_DIR).as_posix() for p in paths]
    sha = [sha256_file(p) for p in paths]
    with open(FIGSHARE_DIR / "labels.csv", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    fig_paths = [FIGSHARE_DIR / row["file"] for row in rows]
    fig_labels = np.array([index[row["label"]] for row in rows])
    fig_patients = [row["patient_id"] for row in rows]
    print(f"Kaggle {len(paths)} images, figshare {len(fig_paths)} images")
    model = load_model()
    emb = embed_files(model, paths, CACHE_DIR / "embeddings", progress=progress)
    fig_emb = embed_files(model, fig_paths, CACHE_DIR / "embeddings", progress=progress)
    del model
    return {
        "paths": paths,
        "rel": rel,
        "labels": labels,
        "sha": sha,
        "emb": emb,
        "fig_emb": fig_emb,
        "fig_labels": fig_labels,
        "fig_patients": fig_patients,
    }


def choose_threshold(rows: list[dict]) -> tuple[float, str]:
    """Applique la règle de décision aux mesures par tranche.

    Args:
        rows: Mesures, du seuil le plus bas au plus haut, avec ``threshold``,
            ``same_patient_rate`` et ``largest_group_share``.

    Returns:
        ``(seuil, explication)``.
    """
    for row in rows:
        rate = row["same_patient_rate"]
        if (
            rate is not None
            and rate >= MIN_SAME_PATIENT
            and row["largest_group_share"] <= MAX_GROUP_SHARE
        ):
            return row["threshold"], "plus basse tranche respectant la règle (a) et (b)"
    return FALLBACK, "aucune tranche ne respecte la règle : seuil de repli"


def main() -> None:
    """Mesure chaque tranche, applique la règle, écrit le JSON et affiche le résumé."""
    start = time.time()
    data = load_inputs()
    emb, labels = data["emb"], data["labels"]
    patients, best = match_patient_ids(
        emb, labels, data["fig_emb"], data["fig_labels"], data["fig_patients"], LINK_COSINE
    )
    known = np.array([bool(p) for p in patients])
    print(
        f"Images Kaggle avec identifiant de patient : {int(known.sum())} "
        f"({len(set(p for p in patients if p))} patients)"
    )
    pairs = similar_pairs(emb, min(TRANCHES))  # toutes les paires >= 0,90, filtrées ensuite
    usable = [(i, j, c) for i, j, c in pairs if known[i] and known[j] and labels[i] == labels[j]]
    print(
        f"Paires >= {min(TRANCHES)} : {len(pairs)} ; "
        f"identifiants connus, même classe : {len(usable)}"
    )

    rows = []
    for threshold in TRANCHES:
        selected = [(i, j) for i, j, c in usable if c >= threshold]
        same = sum(patients[i] == patients[j] for i, j in selected)
        groups = build_groups(emb, labels, threshold, patients, data["sha"])
        groups_no_pid = build_groups(emb, labels, threshold)
        largest = int(np.bincount(groups).max())
        rows.append(
            {
                "threshold": threshold,
                "pairs_known": len(selected),
                "pairs_same_patient": same,
                "same_patient_rate": same / len(selected) if selected else None,
                "n_groups": int(groups.max()) + 1,
                "largest_group": largest,
                "largest_group_share": largest / len(labels),
                "largest_group_without_patient_links": int(np.bincount(groups_no_pid).max()),
            }
        )
        print(
            f"  >= {threshold:.2f} : {len(selected):>6} paires, même patient "
            f"{100 * rows[-1]['same_patient_rate']:.1f} %, groupes {rows[-1]['n_groups']}, "
            f"plus grand {largest} ({100 * largest / len(labels):.1f} %)"
        )

    chosen, why = choose_threshold(rows)
    report = {
        "rule": (
            "plus basse tranche où >= 95 % des paires (de même classe, identifiants connus) sont "
            "du même patient ET plus grand groupe <= 5 % des images ; sinon 0,94"
        ),
        "n_images": len(labels),
        "n_images_with_patient": int(known.sum()),
        "n_patients_found": len({p for p in patients if p}),
        "match_cosine": LINK_COSINE,
        "tranches": rows,
        "chosen_threshold": chosen,
        "reason": why,
    }
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / "patient_threshold_calibration.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Seuil retenu : {chosen} ({why}) ; rapport : {out} ({time.time() - start:.0f} s)")


if __name__ == "__main__":
    main()
