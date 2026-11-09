"""Fabrique le découpage honnête train / val / test de Kaggle (étape 11b-1).

Training et Testing de Kaggle sont FUSIONNÉS puis redécoupés par groupes « même patient »
(voir :mod:`brain_mri.grouping`), car l'ancien découpage fuit : un patient figure des deux côtés.

Étapes :
    1. embeddings DINOv2 de toutes les images Kaggle et des images figshare (cache) ;
    2. identifiants de patient : une image Kaggle copie d'une image figshare (cosinus >= 0,98,
       même classe) reçoit l'identifiant du patient ;
    3. conflits d'étiquette : paires de cosinus >= 0,98 mais de classes différentes ;
       toutes les images concernées sont exclues (planche ``data/cache/label_conflicts.png`` à
       regarder) ;
    4. groupes (union-find : même classe et cosinus >= seuil calibré, même patient, même
       SHA-256) puis ``StratifiedGroupKFold`` à 7 plis, graine 42, strates classe x
       (512 x 512 ou non) : pli 0 = test, pli 1 = validation, 5 plis = entraînement ;
    5. purge : toute image de test dont un voisin d'entraînement (toutes classes) a un cosinus
       >= seuil est retirée du test ;
    6. critères : 0 patient figshare des deux côtés train / test ; 0 image de test avec un
       voisin >= seuil ; au moins 230 images de test par classe.

Le seuil vient de ``docs/figures/patient_threshold_calibration.json`` (script
``calibrate_patient_threshold.py``).

Sorties : ``ml/splits/kaggle_grouped.csv`` (versionné, aucune image : chemin relatif à
``data/raw``, sha256, classe, is_512, groupe, partie, cosinus maximal vers le train) et
``docs/figures/split_report.json``. Les lignes écartées ont pour partie ``conflit`` ou ``purge``.

Utilisation (depuis ``brain-tumor-classifier/ml/``) ::

    uv run python scripts/make_split.py
"""

from __future__ import annotations

import csv
import json
from collections import Counter

import numpy as np
from calibrate_patient_threshold import CLASS_NAMES, FIGURES_DIR, ROOT, load_inputs
from check_overlap import pair_sheet
from PIL import Image

from brain_mri.grouping import (
    LINK_COSINE,
    N_SPLITS,
    SEED,
    build_groups,
    group_split,
    label_conflicts,
    match_patient_ids,
    max_cosine_to,
    residual_leak,
)

SPLITS_DIR = ROOT / "ml" / "splits"
CONFLICT_SHEET = ROOT / "data" / "cache" / "label_conflicts.png"
MIN_TEST_PER_CLASS = 230  # critère : images de test minimales par classe


def count_table(labels: np.ndarray, parts: np.ndarray, mask: np.ndarray | None = None) -> dict:
    """Compte les images par partie et par classe.

    Args:
        labels: Classe (indice) de chaque image.
        parts: Partie de chaque image (``train``, ``val``, ``test``, ``purge``…).
        mask: Images à compter (toutes si ``None``).

    Returns:
        ``{partie: {classe: effectif, "total": effectif}}``.
    """
    mask = np.ones(len(labels), dtype=bool) if mask is None else mask
    table = {}
    for part in sorted(set(parts[mask])):
        selected = mask & (parts == part)
        table[part] = {
            name: int((selected & (labels == i)).sum()) for i, name in enumerate(CLASS_NAMES)
        }
        table[part]["total"] = int(selected.sum())
    return table


def main() -> None:
    """Construit le découpage, vérifie les critères, écrit le manifeste et le rapport."""
    calibration = json.loads(
        (FIGURES_DIR / "patient_threshold_calibration.json").read_text(encoding="utf-8")
    )
    threshold = calibration["chosen_threshold"]
    print(f"Seuil de lien « même patient » : {threshold}")
    data = load_inputs()
    emb, labels, rel, sha = data["emb"], data["labels"], data["rel"], data["sha"]
    n = len(labels)

    # 2. identifiants de patient (copies figshare)
    patients, _ = match_patient_ids(
        emb, labels, data["fig_emb"], data["fig_labels"], data["fig_patients"], LINK_COSINE
    )
    print(f"{sum(bool(p) for p in patients)} images avec identifiant de patient")

    # 3. conflits d'étiquette
    conflicts = label_conflicts(emb, labels, LINK_COSINE)
    conflict_ids = sorted({k for i, j, _ in conflicts for k in (i, j)})
    print(f"{len(conflicts)} paires en conflit, {len(conflict_ids)} images exclues")
    pair_sheet(
        CONFLICT_SHEET,
        [
            (
                data["paths"][i],
                data["paths"][j],
                f"A : {CLASS_NAMES[labels[i]]}  ({rel[i]})"[:44],
                f"B : {CLASS_NAMES[labels[j]]}  cos {c:.3f}",
            )
            for i, j, c in conflicts
        ],
    )
    print(f"Planche des conflits : {CONFLICT_SHEET}")
    keep = np.ones(n, dtype=bool)
    keep[conflict_ids] = False
    kept = np.where(keep)[0]

    # 4. groupes et découpage (sur les images gardées)
    k_emb, k_labels = emb[kept], labels[kept]
    k_patients = [patients[i] for i in kept]
    k_sha = [sha[i] for i in kept]
    groups = build_groups(k_emb, k_labels, threshold, k_patients, k_sha)
    sizes = np.bincount(groups)
    print(
        f"{len(sizes)} groupes, plus grand : {sizes.max()} images "
        f"({100 * sizes.max() / len(kept):.1f} %)"
    )
    is_512 = np.array(
        [Image.open(data["paths"][i]).size == (512, 512) for i in kept], dtype=np.int64
    )
    k_parts = group_split(k_labels, groups, is_512, N_SPLITS, SEED)

    # 5. purge du test
    train = k_parts == "train"
    to_train = max_cosine_to(k_emb, k_emb[train])
    purge = (k_parts == "test") & (to_train >= threshold)
    k_parts = np.where(purge, "purge", k_parts)
    print(f"Purge : {int(purge.sum())} images de test retirées")

    # 6. critères
    test = k_parts == "test"
    patients_by_part = {
        part: {p for p, m in zip(k_patients, k_parts == part, strict=True) if m and p}
        for part in ("train", "val", "test")
    }
    shared = sorted(patients_by_part["train"] & patients_by_part["test"])
    leak_test = residual_leak(k_emb[test], k_emb[k_parts == "train"], threshold)
    leak_val = residual_leak(k_emb[k_parts == "val"], k_emb[k_parts == "train"], threshold)
    test_per_class = {
        name: int((test & (k_labels == i)).sum()) for i, name in enumerate(CLASS_NAMES)
    }
    criteria = {
        "no_figshare_patient_in_train_and_test": {
            "shared_patients": len(shared),
            "patients_train": len(patients_by_part["train"]),
            "patients_test": len(patients_by_part["test"]),
            "passed": not shared,
        },
        "no_test_image_with_train_neighbor": {
            "n_above_threshold": leak_test["n_above_threshold"],
            "passed": leak_test["n_above_threshold"] == 0,
        },
        "at_least_230_test_per_class": {
            "per_class": test_per_class,
            "passed": all(v >= MIN_TEST_PER_CLASS for v in test_per_class.values()),
        },
    }
    for name, c in criteria.items():
        print(f"  critère {name} : {'OK' if c['passed'] else 'ÉCHEC'} {c}")

    # exactitude des doublons exacts : tous dans le même groupe (garanti par le lien SHA-256)
    by_sha: dict[str, set[int]] = {}
    for g, s in zip(groups, k_sha, strict=True):
        by_sha.setdefault(s, set()).add(int(g))
    exact_groups_ok = all(len(v) == 1 for v in by_sha.values())

    # manifeste : toutes les images, y compris conflits et purge (partie explicite)
    parts_all = np.array(["conflit"] * n, dtype=object)
    parts_all[kept] = k_parts
    group_all = np.full(n, -1, dtype=np.int64)
    group_all[kept] = groups
    is512_all = np.array([Image.open(p).size == (512, 512) for p in data["paths"]])
    cos_train = np.full(n, np.nan, dtype=np.float32)
    outside = np.where(parts_all != "train")[0]
    train_emb = emb[parts_all == "train"]
    cos_train[outside] = max_cosine_to(emb[outside], train_emb)
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    manifest = SPLITS_DIR / "kaggle_grouped.csv"
    with open(manifest, "w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["file", "sha256", "class", "is_512", "group", "part", "max_cos_to_train"])
        for i in np.argsort(rel):
            cos = "" if np.isnan(cos_train[i]) else f"{cos_train[i]:.4f}"
            writer.writerow(
                [
                    rel[i],
                    sha[i],
                    CLASS_NAMES[labels[i]],
                    int(is512_all[i]),
                    group_all[i],
                    parts_all[i],
                    cos,
                ]
            )
    print(f"Manifeste : {manifest}")

    counts = count_table(labels, parts_all.astype(str))
    in_split = np.isin(parts_all, ["train", "val", "test"])
    report = {
        "seed": SEED,
        "n_splits": N_SPLITS,
        "threshold": threshold,
        "n_images": n,
        "label_conflicts": {"pairs": len(conflicts), "images_excluded": len(conflict_ids)},
        "groups": {
            "n": len(sizes),
            "largest": int(sizes.max()),
            "largest_share": float(sizes.max() / len(kept)),
            "exact_duplicates_always_same_group": exact_groups_ok,
            "images_with_figshare_patient": int(sum(bool(p) for p in patients)),
        },
        "counts": counts,
        "is_512_by_part_and_class": {
            part: {
                name: {
                    "is_512": int(((parts_all == part) & (labels == i) & is512_all).sum()),
                    "n": int(((parts_all == part) & (labels == i)).sum()),
                }
                for i, name in enumerate(CLASS_NAMES)
            }
            for part in ("train", "val", "test")
        },
        "purge": {
            "n": int(purge.sum()),
            "per_class": dict(Counter(CLASS_NAMES[k_labels[i]] for i in np.where(purge)[0])),
        },
        "residual_leak_test": leak_test,
        "residual_leak_val": leak_val,
        "criteria": criteria,
        "all_criteria_passed": all(c["passed"] for c in criteria.values()),
        "n_in_split": int(in_split.sum()),
    }
    out = FIGURES_DIR / "split_report.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Rapport : {out}")
    for part, row in counts.items():
        print(f"  {part:<8} {row}")


if __name__ == "__main__":
    main()
