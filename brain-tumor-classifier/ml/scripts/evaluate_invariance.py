"""Taux de bascule du modèle sous transformations fixes et sur les paires PMRAM (étape 11b).

Deux mesures, pour un modèle (checkpoint) et un prétraitement (``legacy`` ou ``head_crop``) :

    1. **Transformations fixes** appliquées AVANT le prétraitement du modèle (miroir, bandes
       noires, zoom, recadrage décentré, étirement, JPEG 50) sur une liste d'images : le test
       du manifeste de découpage (``--split-manifest`` ou ``data.split_manifest`` de la
       configuration) ou, à défaut, le dossier ``Testing`` de Kaggle.
    2. **Paires PMRAM** : bascules entre la copie PMRAM et sa jumelle Kaggle (sous-ensemble
       ``seuil_2``, copies sûres). Avec un manifeste, les jumelles en entraînement et en test
       sont séparées. Nécessite les embeddings DINOv2 ; ``--skip-pmram`` saute cette partie.

Utilisation (depuis ``brain-tumor-classifier/ml/``) ::

    uv run python scripts/evaluate_invariance.py --preprocessing head_crop \
        --checkpoint models/head_crop.pt --split-manifest ml/splits/kaggle_grouped.csv

Sortie : un résumé en français et un JSON (``docs/figures/invariance_<prétraitement>.json``
par défaut).
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch

from brain_mri.config import PROJECT_ROOT, Config, load_config
from brain_mri.data import read_split_manifest
from brain_mri.invariance import analyze_pmram_by_twin_part, measure_transformations
from brain_mri.model import build_model_for_checkpoint, get_device, load_checkpoint
from brain_mri.near_duplicates import list_images
from brain_mri.transforms import build_transforms


def pct(value: float | None) -> str:
    """Formate une proportion en pourcentage français (``n/a`` si elle est absente)."""
    return "n/a" if value is None else f"{100 * value:.1f} %".replace(".", ",")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Lit les options de la ligne de commande.

    Args:
        argv: Arguments (``None`` = ceux du programme).

    Returns:
        Les options lues.
    """
    parser = argparse.ArgumentParser(description="Taux de bascule du modèle (étape 11b).")
    parser.add_argument("--config", type=Path, default=None, help="YAML (défaut : train.yaml)")
    parser.add_argument("--checkpoint", type=Path, default=None, help="modèle à mesurer")
    parser.add_argument(
        "--preprocessing", choices=["legacy", "head_crop"], default=None, help="prétraitement"
    )
    parser.add_argument("--split-manifest", type=Path, default=None, help="CSV de découpage")
    parser.add_argument("--max-images", type=int, default=None, help="limite (essais rapides)")
    parser.add_argument("--skip-pmram", action="store_true", help="saute la mesure PMRAM")
    parser.add_argument("--output", type=Path, default=None, help="JSON de sortie")
    return parser.parse_args(argv)


def build_config(args: argparse.Namespace) -> Config:
    """Charge la configuration et applique les options de la ligne de commande.

    Args:
        args: Options lues par :func:`parse_args`.

    Returns:
        La configuration avec le prétraitement et le manifeste demandés.
    """
    config = load_config(args.config) if args.config else load_config()
    update: dict = {}
    if args.preprocessing:
        update["preprocessing"] = args.preprocessing
    if args.split_manifest:
        update["split_manifest"] = args.split_manifest.resolve()
    return config.model_copy(update={"data": config.data.model_copy(update=update)})


def evaluation_images(config: Config, class_names: list[str]) -> tuple[list[Path], np.ndarray]:
    """Liste les images à mesurer : test du manifeste, sinon ``Testing`` de Kaggle.

    Args:
        config: Configuration (``data.split_manifest`` éventuel).
        class_names: Classes du modèle (l'ordre donne l'indice de classe).

    Returns:
        ``(chemins, classes vraies en indices)``.
    """
    index = {name: i for i, name in enumerate(class_names)}
    if config.data.split_manifest is not None:
        samples, _ = read_split_manifest(config.data)
        items = [(path, index[label]) for path, label in samples["test"]]
    else:
        items = [(p, index[p.parent.name]) for p in list_images(config.data.raw_dir / "Testing")]
    return [p for p, _ in items], np.array([i for _, i in items], dtype=np.int64)


def print_transformations(result: dict, class_names: list[str]) -> None:
    """Affiche en français le résultat des transformations fixes.

    Args:
        result: Résultat de ``measure_transformations``.
        class_names: Noms des classes.
    """
    print(
        f"\n=== Transformations fixes : {result['n']} images, "
        f"exactitude {pct(result['base_accuracy'])} ==="
    )
    for name, block in result["transformations"].items():
        o = block["overall"]
        print(
            f"{name:<22} bascules {o['flips']:>4}/{o['n']} ({pct(o['flip_rate'])})  "
            f"juste->faux {o['right_to_wrong']}  faux->juste {o['wrong_to_right']}  "
            f"exactitude {pct(block['accuracy'])}"
        )
        for class_name in class_names:
            c = block["per_class"][class_name]
            print(f"    {class_name:<11} bascules {c['flips']:>4}/{c['n']} ({pct(c['flip_rate'])})")


def pmram_section(
    config: Config, model: torch.nn.Module, class_names: list[str], device: torch.device
) -> dict:
    """Mesure les bascules PMRAM (sous-ensemble ``seuil_2``, copies sûres).

    Args:
        config: Configuration (prétraitement, manifeste éventuel).
        model: Le modèle.
        class_names: Classes du modèle.
        device: CPU ou GPU.

    Returns:
        Le résultat de ``analyze_pmram_by_twin_part``.
    """
    # Imports ici : DINOv2 et les embeddings ne servent qu'à cette mesure.
    from brain_mri.embeddings import embed_files, load_model
    from brain_mri.external_eval import is_duplicate, resolve_path
    from brain_mri.leakage import predict_paths
    from brain_mri.near_duplicates import nearest_cosine

    data_root = config.data.raw_dir.parent
    pm_root = data_root / "external" / "pmram"
    manifest = pm_root / "manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError(f"Manifeste introuvable : {manifest} (lancez download_external.py)")
    entries = json.loads(manifest.read_text(encoding="utf-8"))["images"]
    index = {name: i for i, name in enumerate(class_names)}
    pm_paths = [resolve_path(pm_root, e) for e in entries]
    pm_labels = np.array([index[e["class"]] for e in entries])

    # Pool Kaggle : tout le manifeste (train, val, test) ou bien Training + Testing.
    if config.data.split_manifest is not None:
        samples, _ = read_split_manifest(config.data)
        pool = [(p, index[label], part) for part, items in samples.items() for p, label in items]
    else:
        pool = [
            (p, index[p.parent.name], "train" if "Training" in p.parts else "test")
            for sub in ("Training", "Testing")
            for p in list_images(config.data.raw_dir / sub)
        ]
    kag_paths = [p for p, _, _ in pool]
    kag_labels = np.array([c for _, c, _ in pool])
    kag_part = np.array([part for _, _, part in pool])

    last = [0.0]

    def progress(done: int, count: int) -> None:
        """Affiche l'avancement des embeddings au plus toutes les 30 secondes."""
        if time.time() - last[0] > 30 or done == count:
            last[0] = time.time()
            print(f"  embeddings {done}/{count}", flush=True)

    dino = load_model()
    cache = data_root / "cache" / "embeddings"
    kag_emb = embed_files(dino, kag_paths, cache, progress=progress)
    pm_emb = embed_files(dino, pm_paths, cache, progress=progress)
    del dino
    twin, cosines = nearest_cosine(pm_emb, kag_emb)

    transform = build_transforms(config.data, train=False)
    batch = config.data.batch_size
    copy_pred = predict_paths(model, pm_paths, transform, device, batch)
    unique = sorted(set(twin.tolist()))
    unique_pred = dict(
        zip(
            unique,
            predict_paths(model, [kag_paths[i] for i in unique], transform, device, batch).tolist(),
            strict=True,
        )
    )
    twin_pred = np.array([unique_pred[int(i)] for i in twin])
    far = np.array([not is_duplicate(e) for e in entries])  # sous-ensemble « seuil_2 »
    has_manifest = config.data.split_manifest is not None
    return analyze_pmram_by_twin_part(
        cosines[far],
        (kag_labels[twin] == pm_labels)[far],
        pm_labels[far],
        copy_pred[far],
        twin_pred[far],
        kag_part[twin][far] if has_manifest else None,
        class_names,
    )


def print_pmram(report: dict) -> None:
    """Affiche en français le résultat PMRAM (copies sûres de chaque groupe).

    Args:
        report: Résultat de ``analyze_pmram_by_twin_part``.
    """
    print("\n=== PMRAM (sous-ensemble seuil_2, copies sûres) ===")
    for name, res in report.items():
        s = res["safe"]
        print(
            f"{name:<14} n={s['n']:>4}  bascules {s['flips']} ({pct(s['flip_rate'])})  "
            f"juste->faux {s['right_to_wrong']}  faux->juste {s['wrong_to_right']}  "
            f"jumelle {pct(s['twin_accuracy'])}, copie {pct(s['copy_accuracy'])}"
        )


def main(argv: list[str] | None = None) -> dict:
    """Mesure l'invariance du modèle, affiche le résumé et écrit le JSON.

    Args:
        argv: Arguments de la ligne de commande (``None`` = ceux du programme).

    Returns:
        Le rapport complet (aussi écrit en JSON).
    """
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    args = parse_args(argv)
    config = build_config(args)
    path = args.checkpoint or config.training.checkpoint_path
    if not path.is_file():
        raise FileNotFoundError(f"Modèle introuvable : {path}")

    device = get_device()
    checkpoint = load_checkpoint(path, device)
    class_names = list(checkpoint["class_names"])
    model = build_model_for_checkpoint(checkpoint, config.model)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    mode = config.data.preprocessing
    print(f"Modèle : {path} | prétraitement : {mode} | manifeste : {config.data.split_manifest}")

    paths, labels = evaluation_images(config, class_names)
    if args.max_images:
        paths, labels = paths[: args.max_images], labels[: args.max_images]
    transform = build_transforms(config.data, train=False)
    shown = path.resolve()  # chemin écrit dans le JSON : relatif à la racine du projet si possible
    if shown.is_relative_to(PROJECT_ROOT):
        shown = shown.relative_to(PROJECT_ROOT)
    report: dict = {
        "checkpoint": shown.as_posix(),
        "preprocessing": mode,
        "class_names": class_names,
    }
    report["transformations"] = measure_transformations(
        model, paths, labels, transform, device, class_names, config.data.batch_size
    )
    print_transformations(report["transformations"], class_names)
    if not args.skip_pmram:
        report["pmram"] = pmram_section(config, model, class_names, device)
        print_pmram(report["pmram"])

    out = args.output or config.training.figures_dir / f"invariance_{mode}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)
    print(f"Rapport enregistré dans {out}")
    return report


if __name__ == "__main__":
    torch.manual_seed(0)
    main()
