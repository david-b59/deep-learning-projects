"""Évaluation finale sur le jeu de test (phase D du pipeline).

Le jeu de test n'a servi ni à entraîner ni à choisir le modèle : on le passe **une seule
fois** dans le meilleur modèle (``models/best.pt``) pour obtenir le score honnête, celui
qu'on publie. On produit :

    - l'accuracy globale ;
    - la matrice de confusion (figure et tableau de nombres) ;
    - un rapport par classe : précision, rappel, F1 et nombre d'images ;
    - la liste des images mal classées, pour analyser les erreurs.

Les résultats chiffrés sont écrits dans ``docs/figures/evaluation.json`` et la matrice
dans ``docs/figures/confusion_matrix.png`` (petits fichiers, versionnés pour le README).

Utilisation (depuis ``brain-tumor-classifier/ml/``) ::

    uv run python -m brain_mri.evaluate
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # pas de fenêtre : les figures sont écrites dans des fichiers
import matplotlib.pyplot as plt
import numpy as np
import torch
import truststore
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader

from brain_mri.config import Config, load_config
from brain_mri.data import build_dataloaders
from brain_mri.model import build_model_for_checkpoint
from brain_mri.train import get_device, load_checkpoint

LOGGER = logging.getLogger(__name__)


def predict(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[list[int], list[int]]:
    """Prédit la classe de chaque image du DataLoader, dans l'ordre du jeu de données.

    Le modèle est en mode ``eval()`` et aucun gradient n'est calculé : on mesure, on
    n'apprend pas.

    Args:
        model: Le modèle entraîné.
        loader: Les lots à classer (non mélangés, pour garder l'ordre des fichiers).
        device: CPU ou GPU.

    Returns:
        ``(vraies_classes, classes_prédites)``, deux listes d'indices de même longueur.
    """
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    with torch.no_grad():
        for images, labels in loader:
            logits = model(images.to(device))
            y_pred += logits.argmax(dim=1).cpu().tolist()  # classe au score le plus élevé
            y_true += labels.tolist()
    return y_true, y_pred


def compute_metrics(y_true: list[int], y_pred: list[int], class_names: list[str]) -> dict:
    """Calcule l'accuracy, la matrice de confusion et le rapport par classe.

    Args:
        y_true: Vraies classes (indices).
        y_pred: Classes prédites (indices).
        class_names: Noms des classes, dans l'ordre des indices.

    Returns:
        Un dictionnaire avec ``accuracy``, ``confusion_matrix`` (lignes = vraie classe,
        colonnes = classe prédite), ``per_class`` (précision, rappel, F1, support) et
        ``top_confusions`` (paires les plus confondues, de la plus fréquente à la moins).
    """
    labels = list(range(len(class_names)))
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0,  # classe jamais prédite : précision 0 au lieu d'un avertissement
    )
    per_class = {
        name: {
            "precision": report[name]["precision"],
            "recall": report[name]["recall"],
            "f1": report[name]["f1-score"],
            "support": int(report[name]["support"]),
        }
        for name in class_names
    }
    # Les erreurs sont les cases hors diagonale de la matrice.
    confusions = [
        {"true": class_names[i], "predicted": class_names[j], "count": int(matrix[i, j])}
        for i in labels
        for j in labels
        if i != j and matrix[i, j] > 0
    ]
    confusions.sort(key=lambda c: c["count"], reverse=True)
    return {
        "accuracy": float(np.trace(matrix) / matrix.sum()),  # diagonale / total
        "confusion_matrix": matrix.tolist(),
        "per_class": per_class,
        "top_confusions": confusions,
    }


def list_errors(
    paths: list[str], y_true: list[int], y_pred: list[int], class_names: list[str]
) -> list[dict]:
    """Liste les images mal classées.

    Args:
        paths: Chemin de chaque image, dans le même ordre que ``y_true``.
        y_true: Vraies classes (indices).
        y_pred: Classes prédites (indices).
        class_names: Noms des classes.

    Returns:
        Une entrée par erreur : ``file``, ``true``, ``predicted``.
    """
    return [
        {"file": path, "true": class_names[t], "predicted": class_names[p]}
        for path, t, p in zip(paths, y_true, y_pred, strict=True)
        if t != p
    ]


def plot_confusion_matrix(matrix: list[list[int]], class_names: list[str], path: Path) -> None:
    """Enregistre la matrice de confusion en image (nombres et pourcentage par ligne).

    Args:
        matrix: Matrice de confusion (lignes = vraie classe, colonnes = prédite).
        class_names: Noms des classes.
        path: Fichier PNG à écrire (le dossier est créé si besoin).
    """
    counts = np.array(matrix)
    rows = counts.sum(axis=1, keepdims=True)
    share = counts / np.maximum(rows, 1)  # part de chaque vraie classe, entre 0 et 1

    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(share, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(class_names)), labels=class_names, rotation=30, ha="right")
    ax.set_yticks(range(len(class_names)), labels=class_names)
    ax.set_xlabel("classe prédite")
    ax.set_ylabel("vraie classe")
    ax.set_title("Matrice de confusion (jeu de test)")
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            color = "white" if share[i, j] > 0.5 else "black"  # texte lisible sur fond foncé
            ax.text(
                j, i, f"{counts[i, j]}\n{share[i, j]:.1%}", ha="center", va="center", color=color
            )
    fig.colorbar(image, ax=ax, label="part de la vraie classe")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def evaluate(config: Config) -> dict:
    """Évalue le meilleur modèle sur le jeu de test et écrit les résultats.

    Args:
        config: Configuration complète.

    Returns:
        Le dictionnaire de résultats (accuracy, matrice, rapport, confusions, erreurs).

    Raises:
        FileNotFoundError: Si le fichier du meilleur modèle n'existe pas.
        ValueError: Si les classes du modèle ne correspondent pas à celles des données.
    """
    checkpoint_path = config.training.checkpoint_path
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Modèle introuvable : {checkpoint_path} (lance d'abord train)")

    device = get_device()
    bundle = build_dataloaders(config.data)
    checkpoint = load_checkpoint(checkpoint_path, device)
    if checkpoint["class_names"] != bundle.class_names:
        raise ValueError(
            f"Classes du modèle {checkpoint['class_names']} différentes de celles "
            f"des données {bundle.class_names}"
        )

    # Les poids viennent du fichier : inutile de télécharger ceux d'ImageNet.
    # (le checkpoint indique l'architecture : ResNet-50, ou DINOv2-small gelé + tête linéaire).
    model = build_model_for_checkpoint(checkpoint, config.model).to(device)
    model.load_state_dict(checkpoint["state_dict"])

    LOGGER.info("Évaluation sur le test (%d images)...", bundle.sizes["test"])
    y_true, y_pred = predict(model, bundle.test_loader, device)

    # ImageFolder garde le chemin de chaque image : on le rend relatif pour le rapport.
    samples = bundle.test_loader.dataset.samples
    paths = [Path(p).relative_to(config.data.raw_dir).as_posix() for p, _ in samples]

    results = compute_metrics(y_true, y_pred, bundle.class_names)
    results["errors"] = list_errors(paths, y_true, y_pred, bundle.class_names)
    # « epoch » du checkpoint = numéro DANS la phase : on le renomme pour éviter la confusion
    # avec le numéro global (phase 2, epoch 4 = epoch 7 sur 8).
    results["checkpoint"] = {
        ("epoch_in_phase" if key == "epoch" else key): checkpoint[key]
        for key in ("phase", "epoch", "val_loss", "val_acc")
        if key in checkpoint
    }

    figures_dir = config.training.figures_dir
    figures_dir.mkdir(parents=True, exist_ok=True)
    plot_confusion_matrix(
        results["confusion_matrix"], bundle.class_names, figures_dir / "confusion_matrix.png"
    )
    with open(figures_dir / "evaluation.json", "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False)
    return results


def main() -> None:
    """Point d'entrée en ligne de commande : évalue et affiche un résumé en français."""
    parser = argparse.ArgumentParser(description="Évalue le meilleur modèle sur le jeu de test.")
    parser.add_argument("--config", type=Path, default=None, help="Fichier YAML de configuration")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    truststore.inject_into_ssl()
    config = load_config(args.config) if args.config else load_config()
    results = evaluate(config)

    print(f"\nAccuracy sur le test : {results['accuracy']:.2%}")
    total = sum(map(sum, results["confusion_matrix"]))
    print(f"Erreurs : {len(results['errors'])} image(s) sur {total}")
    print("\nPar classe :")
    for name, scores in results["per_class"].items():
        print(
            f"  {name:<11} précision {scores['precision']:.3f} | rappel {scores['recall']:.3f} "
            f"| F1 {scores['f1']:.3f} | {scores['support']} images"
        )
    print("\nConfusions les plus fréquentes :")
    for c in results["top_confusions"][:5]:
        print(f"  {c['true']} pris pour {c['predicted']} : {c['count']}")


if __name__ == "__main__":
    main()
