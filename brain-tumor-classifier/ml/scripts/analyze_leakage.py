"""Fuite Kaggle et invariance au cadrage du modèle actuel (étape 11a).

Deux analyses, avec les embeddings DINOv2 de la tête recadrée (cache
``data/cache/embeddings/``) et le modèle ``models/best.pt`` :

    a. **Fuite Testing -> Training** : pour chaque image de ``Testing``, plus proche image de
       ``Training`` (cosinus). Nombre d'images avec jumelle (cosinus >= 0,94), part de jumelles
       de même classe, exactitude du modèle avec / sans jumelle, par classe et par tranche de
       cosinus. Mesuré sur toutes les images de ``Testing`` ET sur le jeu de test sans doublons
       exacts (celui de ``evaluate.py``).
    b. **PMRAM, test d'invariance au cadrage** : pour chaque image PMRAM, plus proche image
       Kaggle (Training + Testing). Quand la jumelle est sûre (cosinus >= 0,984, même classe),
       c'est la même image cadrée autrement : on compare la réponse du modèle sur la jumelle
       et sur la copie (« taux de bascule »).

Sorties : ``docs/figures/leakage.json``, un résumé en français, et deux planches dans
``data/cache/`` : ``leakage_pmram_flips.png`` (12 bascules) et
``leakage_testing_training.png`` (18 paires Testing / Training par tranche de cosinus).

Utilisation (depuis ``brain-tumor-classifier/ml/``) ::

    uv run python scripts/analyze_leakage.py
"""

from __future__ import annotations

import json
import logging
import time

import numpy as np
import torch
from check_overlap import pair_sheet

from brain_mri.config import ModelConfig, load_config
from brain_mri.duplicates import load_or_create_report
from brain_mri.embeddings import embed_files, load_model
from brain_mri.external_eval import is_duplicate, resolve_path
from brain_mri.leakage import (
    BIN_LABELS,
    SAFE_COSINE,
    analyze_invariance,
    analyze_kaggle_leak,
    predict_paths,
    select_flips,
    select_pairs_by_bin,
)
from brain_mri.model import build_model, get_device, load_checkpoint
from brain_mri.near_duplicates import DUPLICATE_COSINE, list_images, nearest_cosine
from brain_mri.transforms import build_transforms


def pct(value: float | None) -> str:
    """Formate une proportion en pourcentage français (``n/a`` si elle est absente)."""
    return "n/a" if value is None else f"{100 * value:.1f} %".replace(".", ",")


def print_kaggle(title: str, res: dict) -> None:
    """Affiche en français le résultat de l'analyse Testing -> Training.

    Args:
        title: Titre du jeu mesuré.
        res: Résultat de ``analyze_kaggle_leak``.
    """
    o = res["overall"]
    print(f"\n=== {title} : {o['n']} images ===")
    print(
        f"jumelle dans Training (cos >= {DUPLICATE_COSINE}) : {o['with_twin']} "
        f"({pct(o['with_twin_rate'])}), de même classe : {pct(o['twin_same_class_rate'])}"
    )
    print(
        f"exactitude : {pct(o['accuracy_all'])} ; avec jumelle : "
        f"{pct(o['with_twin_accuracy']['accuracy'])} ({o['with_twin_accuracy']['n']}) ; "
        f"sans jumelle : {pct(o['without_twin_accuracy']['accuracy'])} "
        f"({o['without_twin_accuracy']['n']})"
    )
    for name, b in res["per_class"].items():
        print(
            f"  {name:<11} n={b['n']:>4}  jumelles {b['with_twin']:>4} "
            f"({pct(b['with_twin_rate'])}) "
            f"même classe {pct(b['twin_same_class_rate'])} | exactitude avec "
            f"{pct(b['with_twin_accuracy']['accuracy'])} ({b['with_twin_accuracy']['n']}), sans "
            f"{pct(b['without_twin_accuracy']['accuracy'])} ({b['without_twin_accuracy']['n']})"
        )
    print("  par tranche de cosinus :")
    for label, b in res["by_cosine_bin"].items():
        print(f"    {label:<10} n={b['n']:>4}  exactitude {pct(b['accuracy'])}")


def print_invariance(res: dict) -> None:
    """Affiche en français le résultat de l'analyse PMRAM (invariance au cadrage).

    Args:
        res: Résultat de ``analyze_invariance``.
    """
    s = res["safe"]
    print(f"\n=== PMRAM : {res['n_images']} images, copies sûres (cos >= {res['safe_cosine']}) ===")
    print(
        f"copies sûres de même classe : {s['n']} ; exactitude sur la jumelle Kaggle "
        f"{pct(s['twin_accuracy'])}, sur la copie PMRAM {pct(s['copy_accuracy'])}"
    )
    print(
        f"bascules (réponse différente) : {s['flips']} ({pct(s['flip_rate'])}) ; "
        f"jumelle juste -> copie fausse : {s['right_to_wrong']} ; "
        f"jumelle fausse -> copie juste : {s['wrong_to_right']}"
    )
    for name, b in res["safe_per_class"].items():
        print(
            f"  {name:<11} n={b['n']:>4}  jumelle {pct(b['twin_accuracy'])}, copie "
            f"{pct(b['copy_accuracy'])}, bascules {b['flips']} ({pct(b['flip_rate'])})"
        )
    print("  par tranche de cosinus (jumelle de même classe) :")
    for label, b in res["by_cosine_bin"].items():
        c = b["same_class_twin"]
        print(
            f"    {label:<10} images {b['n_all']:>4}, même classe {c['n']:>4}  jumelle "
            f"{pct(c['twin_accuracy'])}, copie {pct(c['copy_accuracy'])}, "
            f"bascules {c['flips']} ({pct(c['flip_rate'])})"
        )


def main() -> None:
    """Calcule les deux analyses, écrit le JSON et les planches, affiche le résumé."""
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    config = load_config()
    path = config.training.checkpoint_path
    if not path.is_file():
        raise FileNotFoundError(f"Modèle introuvable : {path}")
    data_root = config.data.raw_dir.parent
    pm_root = data_root / "external" / "pmram"
    manifest = pm_root / "manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError(f"Manifeste introuvable : {manifest} (lancez download_external.py)")
    entries = json.loads(manifest.read_text(encoding="utf-8"))["images"]
    cache = data_root / "cache"

    device = get_device()
    checkpoint = load_checkpoint(path, device)
    class_names = list(checkpoint["class_names"])
    model = build_model(len(class_names), ModelConfig(pretrained=False, freeze_backbone=False))
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    transform = build_transforms(config.data, train=False)  # identique au test Kaggle

    train_paths = list_images(config.data.raw_dir / "Training")
    test_paths = list_images(config.data.raw_dir / "Testing")
    pm_paths = [resolve_path(pm_root, e) for e in entries]
    index = {name: i for i, name in enumerate(class_names)}
    train_labels = np.array([index[p.parent.name] for p in train_paths])
    test_labels = np.array([index[p.parent.name] for p in test_paths])
    pm_labels = np.array([index[e["class"]] for e in entries])
    print(f"Training {len(train_paths)}, Testing {len(test_paths)}, PMRAM {len(pm_paths)} images")

    last = [0.0]

    def progress(done: int, count: int) -> None:
        """Affiche l'avancement des embeddings au plus toutes les 30 secondes."""
        if time.time() - last[0] > 30 or done == count:
            last[0] = time.time()
            print(f"  embeddings {done}/{count}", flush=True)

    dino = load_model()
    train_emb = embed_files(dino, train_paths, cache / "embeddings", progress=progress)
    test_emb = embed_files(dino, test_paths, cache / "embeddings", progress=progress)
    pm_emb = embed_files(dino, pm_paths, cache / "embeddings", progress=progress)
    del dino

    # a. Fuite Testing -> Training
    test_twin, test_cos = nearest_cosine(test_emb, train_emb)
    test_pred = predict_paths(model, test_paths, transform, device, config.data.batch_size)
    test_ok = test_pred == test_labels
    same_class = train_labels[test_twin] == test_labels
    excluded = load_or_create_report(config.data.raw_dir, config.data.processed_dir)
    excluded = excluded.files_to_exclude()
    kept = np.array(
        [p.relative_to(config.data.raw_dir).as_posix() not in excluded for p in test_paths]
    )
    report: dict = {"class_names": class_names, "bin_labels": list(BIN_LABELS)}
    report["kaggle_leak"] = {
        "all_testing": analyze_kaggle_leak(test_cos, same_class, test_labels, test_ok, class_names),
        "testing_without_exact_duplicates": analyze_kaggle_leak(
            test_cos[kept], same_class[kept], test_labels[kept], test_ok[kept], class_names
        ),
    }
    print_kaggle("Testing complet (doublons exacts compris)", report["kaggle_leak"]["all_testing"])
    print_kaggle(
        "Testing sans doublons exacts (comme evaluate.py)",
        report["kaggle_leak"]["testing_without_exact_duplicates"],
    )

    # b. PMRAM : copie contre jumelle Kaggle (Training + Testing)
    kag_paths = train_paths + test_paths
    kag_labels = np.concatenate([train_labels, test_labels])
    kag_emb = np.concatenate([train_emb, test_emb])
    pm_twin, pm_cos = nearest_cosine(pm_emb, kag_emb)
    pm_same = kag_labels[pm_twin] == pm_labels
    pm_pred = predict_paths(model, pm_paths, transform, device, config.data.batch_size)
    unique = sorted(set(pm_twin.tolist()))
    unique_pred = dict(
        zip(
            unique,
            predict_paths(
                model, [kag_paths[i] for i in unique], transform, device, config.data.batch_size
            ).tolist(),
            strict=True,
        )
    )
    twin_pred = np.array([unique_pred[int(i)] for i in pm_twin])
    report["pmram_invariance"] = analyze_invariance(
        pm_cos, pm_same, pm_labels, pm_pred, twin_pred, class_names
    )
    print_invariance(report["pmram_invariance"])
    # Même mesure sans les doublons à 2 bits ou moins (identiques à Kaggle : jamais de bascule)
    far = np.array([not is_duplicate(e) for e in entries])
    report["pmram_invariance_seuil_2"] = analyze_invariance(
        pm_cos[far], pm_same[far], pm_labels[far], pm_pred[far], twin_pred[far], class_names
    )
    print("\n--- Sans les doublons à 2 bits ou moins (sous-ensemble « seuil 2 ») ---")
    print_invariance(report["pmram_invariance_seuil_2"])

    # c. Planches
    flips = select_flips(pm_cos, pm_same, pm_pred, twin_pred)
    pair_sheet(
        cache / "leakage_pmram_flips.png",
        [
            (
                pm_paths[i],
                kag_paths[int(pm_twin[i])],
                f"PMRAM ({class_names[pm_labels[i]]}) -> {class_names[pm_pred[i]]}",
                f"Kaggle -> {class_names[twin_pred[i]]}  cos {pm_cos[i]:.3f}",
            )
            for i in flips
        ],
    )
    cells = select_pairs_by_bin(
        test_cos, test_labels, [2, 3, 4], [index["glioma"], index["notumor"]], per_cell=3
    )
    pair_sheet(
        cache / "leakage_testing_training.png",
        [
            (
                test_paths[i],
                train_paths[int(test_twin[i])],
                f"Testing {class_names[test_labels[i]]} [{BIN_LABELS[b]}] "
                f"-> {class_names[test_pred[i]]}",
                f"Training {class_names[train_labels[test_twin[i]]]}  cos {test_cos[i]:.3f}",
            )
            for i, b, _ in cells
        ],
    )
    print(f"\nPlanches : {len(flips)} bascules, {len(cells)} paires Testing/Training (data/cache/)")
    print(f"(seuil des copies sûres : {SAFE_COSINE})")

    out = config.training.figures_dir / "leakage.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)
    print(f"Rapport enregistré dans {out}")


if __name__ == "__main__":
    torch.manual_seed(0)
    main()
