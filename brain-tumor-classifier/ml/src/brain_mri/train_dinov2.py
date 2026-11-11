"""Entraînement du modèle B : DINOv2-small gelé + une couche linéaire (étape 11b-4).

Déroulement :
    1. lecture du manifeste de découpage (``data.split_manifest``) ;
    2. extraction des caractéristiques DINOv2 (768 valeurs par image) avec cache disque :
       - B0 (``dinov2.n_views = 1``) : une vue par image, transformation d'évaluation ;
       - B1 (``dinov2.n_views = K > 1``) : ``K`` vues augmentées par image d'entraînement
         (graine ``seed + k`` pour la vue ``k``) ; validation et test : une vue d'évaluation ;
    3. entraînement de la couche linéaire sur les caractéristiques en mémoire (AdamW, entropie
       croisée, arrêt précoce sur la loss de validation) ;
    4. sauvegarde du meilleur modèle (même format que le ResNet, plus ``architecture``),
       de ``training_history.json`` et de la courbe d'apprentissage.

Le jeu de test n'est ni extrait ni utilisé ici : il sert à l'évaluation finale
(``python -m brain_mri.evaluate --config ...``), qui calcule ses propres caractéristiques.

Utilisation (depuis ``brain-tumor-classifier/ml/``) ::

    uv run python -m brain_mri.train_dinov2 --config configs/train_b0.yaml
    uv run python -m brain_mri.train_dinov2 --config configs/train_b1.yaml
    uv run python -m brain_mri.train_dinov2 --config configs/train_b0.yaml --max-images 40
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import mlflow
import torch
import truststore

from brain_mri.config import Config, load_config
from brain_mri.data import read_split_manifest
from brain_mri.dinov2_head import (
    ARCHITECTURE_NAME,
    CacheMissingError,
    HeadTrainingResult,
    get_features,
    head_state_dict,
    train_head,
)
from brain_mri.embeddings import MODEL_REVISION, load_model
from brain_mri.model import get_device
from brain_mri.train import _flatten, keep_awake, plot_curves, setup_mlflow

LOGGER = logging.getLogger(__name__)


def subsample(items: list, max_images: int | None) -> list:
    """Garde au plus ``max_images`` éléments, régulièrement espacés (essais éclair).

    Args:
        items: Liste complète.
        max_images: Nombre maximal d'éléments (None = tout garder).

    Returns:
        Une liste d'au plus ``max_images`` éléments, répartis sur toute la liste d'origine
        (donc sur plusieurs classes, la liste du manifeste étant triée par chemin).
    """
    if max_images is None or max_images >= len(items):
        return items
    step = len(items) / max_images
    return [items[int(i * step)] for i in range(max_images)]


def save_head_checkpoint(
    path: Path, result: HeadTrainingResult, class_names: list[str], info: dict
) -> None:
    """Enregistre le meilleur modèle B (tête seule) avec le même format que le ResNet.

    Le fichier contient ``state_dict`` (clés ``head.*``), ``class_names``, ``phase``, ``epoch``,
    ``val_loss``, ``val_acc``, ``architecture`` et les informations de ``info``. Le corps DINOv2
    n'y figure pas : :class:`brain_mri.dinov2_head.Dinov2Classifier` le recharge à la révision
    épinglée.

    Args:
        path: Fichier de destination (``.pt``, hors git).
        result: Résultat de l'entraînement de la tête.
        class_names: Noms des classes, dans l'ordre des étiquettes.
        info: Informations supplémentaires (taille d'image, normalisation, prétraitement...).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": head_state_dict(result.head),
        "class_names": class_names,
        "phase": "head",
        "epoch": result.best_epoch,
        "val_loss": result.best_val_loss,
        "val_acc": result.best_val_acc,
        "architecture": ARCHITECTURE_NAME,
        **info,
    }
    torch.save(payload, path)


def train_dinov2(config: Config, max_images: int | None = None, use_mlflow: bool = True) -> dict:
    """Extrait les caractéristiques, entraîne la tête et enregistre les résultats.

    Args:
        config: Configuration complète (``data.split_manifest`` obligatoire).
        max_images: Si fourni, limite chaque partie à ce nombre d'images (essai éclair).
        use_mlflow: Si True, suit l'expérience avec MLflow.

    Returns:
        Un résumé : meilleur epoch, loss et accuracy de validation, et mesures de temps
        d'extraction (``extraction_seconds_per_image``).

    Raises:
        ValueError: Si le manifeste de découpage n'est pas renseigné.
    """
    data_cfg, head_cfg = config.data, config.dinov2
    if data_cfg.split_manifest is None:
        raise ValueError("Le modèle B a besoin de data.split_manifest dans la configuration")
    seed = data_cfg.seed
    device = get_device()
    LOGGER.info("Appareil : %s | prétraitement : %s", device, data_cfg.preprocessing)

    samples, class_names = read_split_manifest(data_cfg)
    index = {name: i for i, name in enumerate(class_names)}
    parts = {
        part: subsample([(path, index[label]) for path, label in items], max_images)
        for part, items in samples.items()
        if part in ("train", "val")  # le test n'est pas touché pendant l'entraînement
    }

    augmented = head_cfg.n_views > 1  # B1 : vues augmentées ; B0 : vue d'évaluation
    plan = {"train": (head_cfg.n_views, augmented), "val": (1, False)}
    backbone = None  # chargé seulement si une vue manque dans le cache

    def get_backbone():
        """Charge DINOv2 (révision épinglée) une seule fois, à la première vue à calculer."""
        nonlocal backbone
        if backbone is None:
            backbone = load_model(device=str(device))
        return backbone

    results = {}
    for part, (n_views, aug) in plan.items():
        # Essai sans corps d'abord : si tout est en cache, DINOv2 n'est même pas chargé.
        try:
            results[part] = get_features(
                None, part, parts[part], class_names, data_cfg, head_cfg, n_views, aug, seed
            )
        except CacheMissingError:
            LOGGER.info("Partie %s : vue(s) absente(s) du cache, chargement de DINOv2", part)
            results[part] = get_features(
                get_backbone(),
                part,
                parts[part],
                class_names,
                data_cfg,
                head_cfg,
                n_views,
                aug,
                seed,
                progress=LOGGER.info,
            )
    computed_images = sum(r.computed_images for r in results.values())
    compute_seconds = sum(r.compute_seconds for r in results.values())
    seconds_per_image = compute_seconds / computed_images if computed_images else None
    if seconds_per_image is not None:
        LOGGER.info(
            "Extraction : %d passage(s) d'image en %.0f s (%.3f s par image)",
            computed_images,
            compute_seconds,
            seconds_per_image,
        )

    train, val = results["train"], results["val"]
    dim = train.features.shape[2]
    train_x = train.features.reshape(-1, dim)  # (N * K, 768) : toutes les vues bout à bout
    train_y = train.labels.repeat_interleave(head_cfg.n_views)  # même ordre que le reshape
    val_x, val_y = val.features[:, 0], val.labels
    LOGGER.info(
        "Tête : %d exemples d'entraînement (%d images x %d vue(s)), %d de validation",
        len(train_x),
        len(train.labels),
        head_cfg.n_views,
        len(val_x),
    )

    def log_epoch(row: dict) -> None:
        """Journalise une epoch (et l'envoie à MLflow si activé) ; une epoch sur 10 est affichée."""
        if row["epoch"] % 10 == 1:
            LOGGER.info(
                "epoch %d | train loss %.4f acc %.3f | val loss %.4f acc %.3f",
                row["epoch"],
                row["train_loss"],
                row["train_acc"],
                row["val_loss"],
                row["val_acc"],
            )
        if use_mlflow:
            mlflow.log_metrics(
                {k: v for k, v in row.items() if isinstance(v, float)}, step=row["epoch"]
            )

    def run() -> HeadTrainingResult:
        """Entraîne la tête, écrit le checkpoint, l'historique et la courbe."""
        start = time.time()
        result = train_head(
            train_x, train_y, val_x, val_y, len(class_names), head_cfg, seed, on_epoch=log_epoch
        )
        LOGGER.info("Tête entraînée en %.1f s", time.time() - start)
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
                "n_views": head_cfg.n_views,
            },
        )
        history_path = data_cfg.processed_dir / "training_history.json"
        history_path.parent.mkdir(parents=True, exist_ok=True)
        history_path.write_text(json.dumps(result.history, indent=2), encoding="utf-8")
        curves = config.training.figures_dir / "training_curves.png"
        plot_curves(result.history, curves)
        if use_mlflow:
            mlflow.log_artifact(str(curves))
            mlflow.log_metric("best_val_loss", result.best_val_loss)
            mlflow.log_metric("best_val_acc", result.best_val_acc)
            if seconds_per_image is not None:
                mlflow.log_metric("extraction_seconds_per_image", seconds_per_image)
        return result

    if use_mlflow:
        setup_mlflow(config)
        with mlflow.start_run():
            mlflow.log_params(_flatten("config", json.loads(config.model_dump_json())))
            result = run()
    else:
        result = run()
    LOGGER.info(
        "Terminé : meilleur epoch %d, val loss %.4f, val acc %.4f",
        result.best_epoch,
        result.best_val_loss,
        result.best_val_acc,
    )
    return {
        "best_epoch": result.best_epoch,
        "best_val_loss": result.best_val_loss,
        "best_val_acc": result.best_val_acc,
        "extraction_seconds_per_image": seconds_per_image,
        "epochs_run": len(result.history),
    }


def main() -> None:
    """Point d'entrée en ligne de commande."""
    parser = argparse.ArgumentParser(description="Entraîne le modèle B (DINOv2 gelé + tête).")
    parser.add_argument("--config", type=Path, required=True, help="fichier YAML de configuration")
    parser.add_argument(
        "--max-images", type=int, default=None, help="images par partie (essai éclair)"
    )
    parser.add_argument("--no-mlflow", action="store_true", help="désactive le suivi MLflow")
    parser.add_argument(
        "--keep-awake", action="store_true", help="empêche la mise en veille de Windows"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    truststore.inject_into_ssl()  # téléchargement de DINOv2 derrière un proxy
    config = load_config(args.config)
    with keep_awake(args.keep_awake):
        summary = train_dinov2(config, max_images=args.max_images, use_mlflow=not args.no_mlflow)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
