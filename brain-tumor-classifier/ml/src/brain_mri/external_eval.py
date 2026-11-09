"""Mesure du modèle sur un jeu externe (PMRAM), étape 11a.

Le jeu externe est décrit par un manifeste (``manifest.json``, clé ``images``). Les images
sont lues depuis leur dossier réel (``<classe>/`` si elles sont gardées, ``_excluded/``
sinon) avec EXACTEMENT le même prétraitement que le test Kaggle
(``build_transforms(train=False)``). On les répartit en trois sous-ensembles :

    - ``strict``   : statut ``kept`` (aucune image proche de Kaggle au seuil de 6 bits) ;
    - ``seuil_2``  : pas de jumelle Kaggle à 2 bits ou moins (distance absente ou > 2) ;
    - ``doublons`` : jumelle Kaggle à 2 bits ou moins (contrôle : le modèle doit y exceller).
"""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

from brain_mri.evaluate import compute_metrics

EXCLUDED_DIR_NAME = "_excluded"  # dossier des images écartées par le script de téléchargement
DUPLICATE_MAX_DISTANCE = 2  # au plus 2 bits d'écart dHash = vrai doublon
NOTUMOR = "notumor"  # nom de la classe « pas de tumeur »
SUBSET_NAMES = ("strict", "seuil_2", "doublons")


def is_duplicate(entry: dict) -> bool:
    """Dit si l'image a une jumelle Kaggle à 2 bits ou moins.

    Args:
        entry: Entrée du manifeste.

    Returns:
        True si la distance existe et vaut 2 ou moins.
    """
    distance = entry.get("distance")
    return distance is not None and distance <= DUPLICATE_MAX_DISTANCE


def select_subsets(entries: list[dict]) -> dict[str, list[dict]]:
    """Répartit les entrées du manifeste en trois sous-ensembles.

    Args:
        entries: Liste ``images`` du manifeste (``status`` et ``distance`` utilisés).

    Returns:
        Un dictionnaire ``nom -> entrées`` pour ``strict``, ``seuil_2`` et ``doublons``.
    """
    return {
        "strict": [e for e in entries if e["status"] == "kept"],
        "seuil_2": [e for e in entries if not is_duplicate(e)],
        "doublons": [e for e in entries if is_duplicate(e)],
    }


def resolve_path(root: Path, entry: dict) -> Path:
    """Donne le chemin réel d'une image du manifeste.

    Selon le jeu, les images écartées sont déplacées dans ``_excluded/`` (PMRAM) ou
    laissées à leur place (BDNeuro : statuts ``excluded_copy`` et ``gray_zone``). On cherche
    donc d'abord à l'emplacement attendu d'après le statut, puis à l'autre.

    Args:
        root: Dossier du jeu externe.
        entry: Entrée du manifeste.

    Returns:
        Le chemin qui existe ; si aucun n'existe, l'emplacement attendu d'après le statut
        (``root/file`` pour une image gardée, ``root/_excluded/file`` sinon).
    """
    in_place = root / entry["file"]
    moved = root / EXCLUDED_DIR_NAME / entry["file"]
    expected, other = (in_place, moved) if entry["status"] == "kept" else (moved, in_place)
    if expected.is_file() or not other.is_file():
        return expected
    return other


class ExternalDataset(Dataset):
    """Jeu d'images externes : renvoie ``(tenseur, indice de classe)``."""

    def __init__(
        self,
        root: Path,
        entries: list[dict],
        class_names: list[str],
        transform: v2.Compose,
    ) -> None:
        """Prépare la liste des fichiers.

        Args:
            root: Dossier du jeu externe.
            entries: Entrées du manifeste à charger.
            class_names: Noms des classes du modèle, dans l'ordre des indices.
            transform: Prétraitement (celui du test Kaggle).
        """
        self.paths = [resolve_path(root, e) for e in entries]
        self.labels = [class_names.index(e["class"]) for e in entries]
        self.transform = transform

    def __len__(self) -> int:
        """Nombre d'images."""
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        """Charge l'image ``index`` en RGB et applique le prétraitement."""
        with Image.open(self.paths[index]) as image:
            return self.transform(image.convert("RGB")), self.labels[index]


def predict_with_confidence(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[list[int], list[int], list[float]]:
    """Prédit la classe et la confiance (probabilité maximale) de chaque image.

    Args:
        model: Le modèle entraîné.
        loader: Les lots à classer (non mélangés).
        device: CPU ou GPU.

    Returns:
        ``(vraies_classes, classes_prédites, confiances)``.
    """
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    confidences: list[float] = []
    with torch.no_grad():
        for images, labels in loader:
            probs = torch.softmax(model(images.to(device)), dim=1)  # scores -> probabilités
            best, pred = probs.max(dim=1)
            y_pred += pred.cpu().tolist()
            confidences += best.cpu().tolist()
            y_true += labels.tolist()
    return y_true, y_pred, confidences


def _mean(values: list[float]) -> float | None:
    """Moyenne d'une liste, ou ``None`` si elle est vide."""
    return sum(values) / len(values) if values else None


def summarize(
    y_true: list[int], y_pred: list[int], confidences: list[float], class_names: list[str]
) -> dict:
    """Résume les prédictions d'un sous-ensemble.

    Args:
        y_true: Vraies classes (indices).
        y_pred: Classes prédites (indices).
        confidences: Probabilité maximale de chaque prédiction.
        class_names: Noms des classes.

    Returns:
        Effectifs par classe, exactitude, rappel par classe, matrice de confusion, taux de
        tumeurs prédites « notumor », nombre de « notumor » prédites tumeur, confiance
        moyenne des bonnes et des mauvaises réponses (``None`` si aucune).
    """
    result: dict = {"n": len(y_true)}
    if not y_true:
        return result
    metrics = compute_metrics(y_true, y_pred, class_names)
    notumor = class_names.index(NOTUMOR)
    pairs = list(zip(y_true, y_pred, confidences, strict=True))
    tumor_preds = [p for t, p, _ in pairs if t != notumor]
    healthy_preds = [p for t, p, _ in pairs if t == notumor]
    missed = sum(p == notumor for p in tumor_preds)
    result.update(
        {
            "per_class_n": {n: metrics["per_class"][n]["support"] for n in class_names},
            "accuracy": metrics["accuracy"],
            "recall": {n: metrics["per_class"][n]["recall"] for n in class_names},
            "confusion_matrix": metrics["confusion_matrix"],
            "tumor_predicted_notumor": {
                "count": missed,
                "total": len(tumor_preds),
                "rate": missed / len(tumor_preds) if tumor_preds else None,
            },
            "notumor_predicted_tumor": {
                "count": sum(p != notumor for p in healthy_preds),
                "total": len(healthy_preds),
            },
            "mean_confidence_correct": _mean([c for t, p, c in pairs if t == p]),
            "mean_confidence_wrong": _mean([c for t, p, c in pairs if t != p]),
        }
    )
    return result


def evaluate_subset(
    model: nn.Module,
    root: Path,
    entries: list[dict],
    class_names: list[str],
    transform: v2.Compose,
    device: torch.device,
    batch_size: int = 32,
) -> dict:
    """Mesure le modèle sur un sous-ensemble du manifeste.

    Args:
        model: Le modèle entraîné.
        root: Dossier du jeu externe.
        entries: Entrées du sous-ensemble.
        class_names: Noms des classes du modèle.
        transform: Prétraitement (celui du test Kaggle).
        device: CPU ou GPU.
        batch_size: Taille des lots.

    Returns:
        Le résumé de :func:`summarize`.
    """
    if not entries:
        return {"n": 0}
    dataset = ExternalDataset(root, entries, class_names, transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    y_true, y_pred, conf = predict_with_confidence(model, loader, device)
    return summarize(y_true, y_pred, conf, class_names)
