"""DFR : Deep Feature Reweighting sur DINOv2 gelé (étape 11b-6b, Kirichenko et al. 2023).

Idée : dans Kaggle, la source (image 512 x 512 ou non) trahit presque la classe. Une tête
linéaire entraînée normalement peut apprendre ce raccourci. Ici on réentraîne SEULEMENT la tête
linéaire sur les caractéristiques DINOv2 gelées (8 vues augmentées en cache, comme B1), en
donnant le même poids total à chaque groupe classe x source
(``brain_mri.source_bias.group_weights``).

Déroulement :
    1. lecture du manifeste (classe, source ``is_512``, groupe patient) des images de ``train`` ;
    2. VALIDATION CROISÉE GROUPÉE à 5 plis (``StratifiedGroupKFold``, groupes = patients, graine
       42) pour choisir le ``weight_decay`` dans une grille. Pour chaque pli : la tête est
       entraînée sur les 4 autres plis (poids recalculés sur ces images), la validation du jeu
       ``val`` sert à l'arrêt précoce (comme B1), puis on prédit les images du pli retenu sur
       leur vue d'ÉVALUATION. Critère : exactitude du PIRE groupe classe x source (groupes de
       moins de 5 images ignorés dans ce minimum), calculée sur toutes les prédictions hors pli ;
    3. les prédictions hors pli au ``weight_decay`` retenu donnent le critère S3 (exactitude sur
       les tumeurs hors 512 du train) ;
    4. entraînement final sur tout le train avec les poids de groupe, même format de point de
       contrôle que B1 (``models/b_dfr/best.pt``), historique et courbe.

Sortie de la validation croisée : ``docs/figures/b_dfr/cv.json``. Le test n'est pas touché ici
(il se mesure avec ``brain_mri.evaluate`` et ``scripts/evaluate_invariance.py``).

Utilisation (depuis ``brain-tumor-classifier/ml/``) ::

    PYTHONUTF8=1 uv run python scripts/train_dfr.py --config configs/train_b_dfr.yaml
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedGroupKFold

from brain_mri.config import PROJECT_ROOT, Config, load_config
from brain_mri.data import read_split_manifest
from brain_mri.dinov2_head import get_features, train_head
from brain_mri.embeddings import MODEL_REVISION
from brain_mri.model_selection import NOTUMOR, bootstrap_mean
from brain_mri.source_bias import group_weights
from brain_mri.train import plot_curves
from brain_mri.train_dinov2 import save_head_checkpoint

LOGGER = logging.getLogger("train_dfr")

DEFAULT_GRID = (0.0, 1e-4, 1e-3, 1e-2, 1e-1)  # valeurs de weight_decay essayées
N_FOLDS = 5  # plis de la validation croisée groupée
MIN_GROUP_SIZE = 5  # un groupe plus petit est ignoré dans le minimum du pire groupe


def load_train_metadata(config: Config) -> tuple[np.ndarray, np.ndarray]:
    """Lit, pour les images ``train`` du manifeste, l'indicateur 512 et le groupe patient.

    Args:
        config: Configuration (``data.split_manifest`` obligatoire).

    Returns:
        ``(is_512, groups)`` dans l'ordre du manifeste (celui de ``read_split_manifest``).
    """
    data = config.data
    with open(data.split_manifest, encoding="utf-8", newline="") as file:
        rows = [r for r in csv.DictReader(file) if r[data.manifest_split_column].strip() == "train"]
    is_512 = np.array([r["is_512"] == "1" for r in rows])
    groups = np.array([r["group"] for r in rows])
    return is_512, groups


def worst_group_accuracy(
    labels: np.ndarray, predictions: np.ndarray, is_512: np.ndarray
) -> tuple[float, dict[str, dict]]:
    """Exactitude du pire groupe classe x source (groupes de moins de 5 images ignorés).

    Args:
        labels: Vraies classes, ``(N,)``.
        predictions: Classes prédites, ``(N,)``.
        is_512: Indicateur « 512 x 512 », ``(N,)``.

    Returns:
        ``(minimum, détail)`` ; le détail donne ``n`` et ``accuracy`` de CHAQUE groupe
        (clé ``classe_indice/512`` ou ``classe_indice/autre``), y compris les petits groupes.
    """
    detail: dict[str, dict] = {}
    scores = []
    for label in np.unique(labels):
        for source in (True, False):
            mask = (labels == label) & (is_512 == source)
            if not mask.any():
                continue
            accuracy = float((predictions[mask] == labels[mask]).mean())
            detail[f"{label}/{'512' if source else 'autre'}"] = {
                "n": int(mask.sum()),
                "accuracy": accuracy,
                "ignored": bool(mask.sum() < MIN_GROUP_SIZE),
            }
            if mask.sum() >= MIN_GROUP_SIZE:
                scores.append(accuracy)
    return min(scores), detail


def cross_validate(
    train_x: torch.Tensor,
    eval_x: torch.Tensor,
    labels: np.ndarray,
    is_512: np.ndarray,
    groups: np.ndarray,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    config: Config,
    n_classes: int,
    grid: tuple[float, ...],
    seed: int,
) -> dict:
    """Validation croisée groupée : prédictions hors pli pour chaque ``weight_decay``.

    Args:
        train_x: Caractéristiques augmentées ``(N, n_views, d)``.
        eval_x: Caractéristiques de la vue d'évaluation ``(N, d)``.
        labels: Classes du train, ``(N,)``.
        is_512: Indicateur 512 du train, ``(N,)``.
        groups: Groupes patients du train, ``(N,)``.
        val_x: Caractéristiques de validation (arrêt précoce), ``(V, d)``.
        val_y: Classes de validation, ``(V,)``.
        config: Configuration (réglages de la tête).
        n_classes: Nombre de classes.
        grid: Valeurs de ``weight_decay`` à essayer.
        seed: Graine (plis et entraînement).

    Returns:
        ``{"folds": [...], "grid": {wd: {worst_group_accuracy, groups, overall_accuracy,
        predictions}}}`` ; ``predictions`` (liste d'indices) sert au critère S3.
    """
    n_views = train_x.shape[1]
    splitter = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    folds = list(splitter.split(np.zeros(len(labels)), labels, groups))
    results: dict[str, dict] = {}
    for wd in grid:
        head_cfg = config.dinov2.model_copy(update={"weight_decay": wd})
        predictions = np.full(len(labels), -1)
        start = time.time()
        for fold, (fit_idx, held_idx) in enumerate(folds, start=1):
            x = train_x[fit_idx].reshape(-1, train_x.shape[2])  # toutes les vues bout à bout
            y = torch.from_numpy(labels[fit_idx]).repeat_interleave(n_views)
            w = torch.from_numpy(group_weights(labels[fit_idx], is_512[fit_idx])).float()
            result = train_head(
                x,
                y,
                val_x,
                val_y,
                n_classes,
                head_cfg,
                seed,
                sample_weights=w.repeat_interleave(n_views),
            )
            with torch.no_grad():
                predictions[held_idx] = result.head(eval_x[held_idx]).argmax(dim=1).numpy()
            LOGGER.info("wd=%g pli %d/%d : meilleur epoch %d", wd, fold, N_FOLDS, result.best_epoch)
        worst, detail = worst_group_accuracy(labels, predictions, is_512)
        results[repr(wd)] = {
            "weight_decay": wd,
            "worst_group_accuracy": worst,
            "groups": detail,
            "overall_accuracy": float((predictions == labels).mean()),
            "predictions": predictions.tolist(),
        }
        LOGGER.info(
            "wd=%g : pire groupe %.4f, exactitude %.4f (%.0f s)",
            wd,
            worst,
            results[repr(wd)]["overall_accuracy"],
            time.time() - start,
        )
    return {
        "folds": [{"fold": i, "n_held_out": len(h)} for i, (_, h) in enumerate(folds, start=1)],
        "grid": results,
    }


def main() -> None:
    """Point d'entrée : validation croisée, choix du ``weight_decay``, entraînement final."""
    parser = argparse.ArgumentParser(description="DFR sur DINOv2 gelé (étape 11b-6b).")
    parser.add_argument("--config", type=Path, required=True, help="fichier YAML de configuration")
    parser.add_argument(
        "--grid", type=float, nargs="+", default=list(DEFAULT_GRID), help="valeurs de weight_decay"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    config = load_config(args.config)
    data_cfg, head_cfg = config.data, config.dinov2
    seed = data_cfg.seed
    samples, class_names = read_split_manifest(data_cfg)
    index = {name: i for i, name in enumerate(class_names)}
    parts = {p: [(path, index[label]) for path, label in samples[p]] for p in ("train", "val")}
    is_512, groups = load_train_metadata(config)
    if len(is_512) != len(parts["train"]):
        raise ValueError("Le nombre d'images train du manifeste ne correspond pas")

    # Tout est lu dans le cache disque (corps None) : aucune extraction ici.
    train = get_features(
        None, "train", parts["train"], class_names, data_cfg, head_cfg, head_cfg.n_views, True, seed
    )
    train_eval = get_features(
        None, "train", parts["train"], class_names, data_cfg, head_cfg, 1, False, seed
    )
    val = get_features(None, "val", parts["val"], class_names, data_cfg, head_cfg, 1, False, seed)
    labels = train.labels.numpy()
    val_x, val_y = val.features[:, 0], val.labels
    n_classes = len(class_names)
    LOGGER.info(
        "Train : %d images x %d vues ; hors 512 : %d ; val : %d",
        len(labels),
        head_cfg.n_views,
        int((~is_512).sum()),
        len(val_y),
    )

    cv = cross_validate(
        train.features,
        train_eval.features[:, 0],
        labels,
        is_512,
        groups,
        val_x,
        val_y,
        config,
        n_classes,
        tuple(args.grid),
        seed,
    )
    best_key = max(  # pire groupe, puis exactitude globale en cas d'égalité
        cv["grid"],
        key=lambda k: (cv["grid"][k]["worst_group_accuracy"], cv["grid"][k]["overall_accuracy"]),
    )
    best = cv["grid"][best_key]
    wd = best["weight_decay"]
    notumor = class_names.index(NOTUMOR)
    target = (labels != notumor) & ~is_512  # tumeurs hors 512 du train
    oof_correct = np.asarray(best["predictions"])[target] == labels[target]
    cv_report = {
        "criterion": f"exactitude du pire groupe classe x source (groupes < {MIN_GROUP_SIZE} "
        "images ignorés), prédictions hors pli",
        "class_names": class_names,
        "n_folds": N_FOLDS,
        "seed": seed,
        "folds": cv["folds"],
        "grid": {
            k: {key: v[key] for key in v if key != "predictions"} for k, v in cv["grid"].items()
        },
        "selected_weight_decay": wd,
        "s3_tumors_not_512_oof": {
            "n": int(target.sum()),
            "n_correct": int(oof_correct.sum()),
            "accuracy": bootstrap_mean(oof_correct.astype(np.int64)),
        },
    }
    out_dir = config.training.figures_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "cv.json").write_text(
        json.dumps(cv_report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    LOGGER.info(
        "weight_decay retenu : %g ; S3 hors pli : %s", wd, cv_report["s3_tumors_not_512_oof"]
    )

    # Entraînement final sur tout le train, poids de groupe, val comme pour B1.
    final_cfg = head_cfg.model_copy(update={"weight_decay": wd})
    n_views = head_cfg.n_views
    x = train.features.reshape(-1, train.features.shape[2])
    y = train.labels.repeat_interleave(n_views)
    w = torch.from_numpy(group_weights(labels, is_512)).float().repeat_interleave(n_views)
    result = train_head(x, y, val_x, val_y, n_classes, final_cfg, seed, sample_weights=w)
    save_head_checkpoint(
        config.training.checkpoint_path,
        result,
        class_names,
        {
            "image_size": data_cfg.image_size,
            "mean": data_cfg.mean,
            "std": data_cfg.std,
            "preprocessing": data_cfg.preprocessing,
            "dinov2_revision": MODEL_REVISION,
            "n_views": n_views,
        },
    )
    history_path = data_cfg.processed_dir / "training_history.json"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(json.dumps(result.history, indent=2), encoding="utf-8")
    plot_curves(result.history, out_dir / "training_curves.png")
    LOGGER.info(
        "Terminé : meilleur epoch %d, val loss %.4f, val acc %.4f (checkpoint %s)",
        result.best_epoch,
        result.best_val_loss,
        result.best_val_acc,
        config.training.checkpoint_path.relative_to(PROJECT_ROOT).as_posix(),
    )


if __name__ == "__main__":
    main()
