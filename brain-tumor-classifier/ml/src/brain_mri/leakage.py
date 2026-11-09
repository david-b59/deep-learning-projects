"""Analyses de fuite et d'invariance au cadrage (étape 11a).

Deux questions, une même mesure (le cosinus DINOv2 de la tête recadrée, voir
:mod:`brain_mri.embeddings`) :

    a. **Fuite Kaggle** : beaucoup d'images de ``Testing`` ont une « jumelle » dans
       ``Training`` (cosinus >= 0,94). Le modèle est-il meilleur sur elles que sur les
       images sans jumelle ?
    b. **Invariance au cadrage** : une image PMRAM dont la jumelle Kaggle est très sûre
       (cosinus >= 0,984, même classe) est la MÊME image, recadrée ou zoomée autrement. Si le
       modèle change de réponse entre la jumelle et la copie, il dépend du cadrage.

Le module ne contient que de la logique pure (tableaux numpy) plus une fonction de
prédiction ; le script ``scripts/analyze_leakage.py`` fournit les fichiers, les embeddings
et le modèle réel.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision.transforms import v2

from brain_mri.near_duplicates import DUPLICATE_COSINE

SAFE_COSINE = 0.984  # au-dessus, la jumelle Kaggle d'une image PMRAM est jugée certaine
BIN_EDGES = (0.90, 0.94, 0.96, 0.98)  # bornes des tranches de cosinus
BIN_LABELS = ("<0,90", "0,90-0,94", "0,94-0,96", "0,96-0,98", ">=0,98")  # une de plus que de bornes


def assign_bins(cosines: np.ndarray) -> np.ndarray:
    """Donne l'indice de tranche de chaque cosinus (borne basse incluse).

    Args:
        cosines: Cosinus, tableau ``(N,)``.

    Returns:
        Entiers de 0 (``<0,90``) à 4 (``>=0,98``).
    """
    return np.digitize(cosines, BIN_EDGES)


def accuracy_summary(correct: np.ndarray, mask: np.ndarray) -> dict:
    """Effectif et exactitude d'un groupe d'images.

    Args:
        correct: Booléens « le modèle a juste », ``(N,)``.
        mask: Booléens « l'image est dans le groupe », ``(N,)``.

    Returns:
        ``{"n": effectif, "accuracy": exactitude ou None si le groupe est vide}``.
    """
    n = int(mask.sum())
    return {"n": n, "accuracy": float(correct[mask].mean()) if n else None}


def predict_paths(
    model: nn.Module,
    paths: Sequence[Path],
    transform: v2.Compose,
    device: torch.device,
    batch_size: int = 32,
) -> np.ndarray:
    """Prédit la classe de chaque fichier image (même prétraitement que le test).

    Args:
        model: Le modèle entraîné.
        paths: Chemins des images.
        transform: Prétraitement (``build_transforms(train=False)``).
        device: CPU ou GPU.
        batch_size: Taille des lots.

    Returns:
        Indices de classe prédits, tableau d'entiers ``(len(paths),)``.
    """
    model.eval()
    predictions: list[int] = []
    with torch.no_grad():
        for start in range(0, len(paths), batch_size):
            tensors = []
            for path in paths[start : start + batch_size]:
                with Image.open(path) as image:
                    tensors.append(transform(image.convert("RGB")))
            scores = model(torch.stack(tensors).to(device))
            predictions += scores.argmax(dim=1).cpu().tolist()
    return np.array(predictions, dtype=np.int64)


def analyze_kaggle_leak(
    cosines: np.ndarray,
    twin_same_class: np.ndarray,
    labels: np.ndarray,
    correct: np.ndarray,
    class_names: list[str],
    threshold: float = DUPLICATE_COSINE,
) -> dict:
    """Mesure la fuite Testing -> Training et son effet sur l'exactitude.

    Args:
        cosines: Pour chaque image de test, cosinus avec sa plus proche image de Training.
        twin_same_class: Booléens « cette plus proche image est de la même classe ».
        labels: Vraie classe de chaque image de test (indice).
        correct: Booléens « le modèle a juste ».
        class_names: Noms des classes.
        threshold: Cosinus à partir duquel on parle de jumelle.

    Returns:
        Résultats global, par classe et par tranche de cosinus : effectifs, part avec jumelle,
        part de jumelles de même classe, exactitude avec et sans jumelle.
    """
    has_twin = cosines >= threshold

    def block(mask: np.ndarray) -> dict:
        """Résumé d'un groupe (toutes les images ou une classe)."""
        twins = mask & has_twin
        n_twins = int(twins.sum())
        return {
            "n": int(mask.sum()),
            "with_twin": n_twins,
            "with_twin_rate": n_twins / int(mask.sum()) if mask.any() else None,
            "twin_same_class_rate": float(twin_same_class[twins].mean()) if n_twins else None,
            "accuracy_all": accuracy_summary(correct, mask)["accuracy"],
            "with_twin_accuracy": accuracy_summary(correct, twins),
            "without_twin_accuracy": accuracy_summary(correct, mask & ~has_twin),
        }

    bins = assign_bins(cosines)
    by_bin = {}
    for index, label in enumerate(BIN_LABELS):
        in_bin = bins == index
        by_bin[label] = {
            **accuracy_summary(correct, in_bin),
            "per_class": {
                name: accuracy_summary(correct, in_bin & (labels == i))
                for i, name in enumerate(class_names)
            },
        }
    return {
        "threshold": threshold,
        "overall": block(np.ones(len(cosines), dtype=bool)),
        "per_class": {name: block(labels == i) for i, name in enumerate(class_names)},
        "by_cosine_bin": by_bin,
    }


def analyze_invariance(
    cosines: np.ndarray,
    twin_same_class: np.ndarray,
    labels: np.ndarray,
    copy_pred: np.ndarray,
    twin_pred: np.ndarray,
    class_names: list[str],
    safe_cosine: float = SAFE_COSINE,
) -> dict:
    """Compare le modèle sur des copies PMRAM et sur leurs jumelles Kaggle.

    Une « bascule » est une image pour laquelle le modèle donne une réponse différente sur la
    copie et sur la jumelle. Les jumelles retenues sont celles de MÊME classe (la vérité est
    alors la même des deux côtés).

    Args:
        cosines: Pour chaque image PMRAM, cosinus avec sa plus proche image Kaggle.
        twin_same_class: Booléens « la jumelle Kaggle est de la même classe ».
        labels: Vraie classe de chaque image PMRAM (indice).
        copy_pred: Classe prédite sur la copie PMRAM.
        twin_pred: Classe prédite sur la jumelle Kaggle.
        class_names: Noms des classes.
        safe_cosine: Cosinus minimal des « copies sûres ».

    Returns:
        Résultats des copies sûres (global et par classe) et par tranche de cosinus.
    """
    twin_ok = twin_pred == labels
    copy_ok = copy_pred == labels
    flip = copy_pred != twin_pred

    def block(mask: np.ndarray) -> dict:
        """Résumé d'un groupe d'images (jumelle de même classe déjà imposée par l'appelant)."""
        n = int(mask.sum())
        return {
            "n": n,
            "twin_accuracy": float(twin_ok[mask].mean()) if n else None,
            "copy_accuracy": float(copy_ok[mask].mean()) if n else None,
            "flips": int(flip[mask].sum()),
            "flip_rate": float(flip[mask].mean()) if n else None,
            "right_to_wrong": int((twin_ok & ~copy_ok)[mask].sum()),  # jumelle juste, copie fausse
            "wrong_to_right": int((~twin_ok & copy_ok)[mask].sum()),
        }

    same = twin_same_class
    safe = same & (cosines >= safe_cosine)
    bins = assign_bins(cosines)
    return {
        "safe_cosine": safe_cosine,
        "n_images": len(cosines),
        "safe": block(safe),
        "safe_per_class": {name: block(safe & (labels == i)) for i, name in enumerate(class_names)},
        "by_cosine_bin": {
            label: {
                "n_all": int((bins == index).sum()),
                "same_class_twin": block(same & (bins == index)),
            }
            for index, label in enumerate(BIN_LABELS)
        },
    }


def sample_indices(candidates: Sequence[int], count: int, seed: int = 42) -> list[int]:
    """Tire ``count`` indices au hasard (graine fixe), rendus triés.

    Args:
        candidates: Indices possibles.
        count: Nombre à tirer (au plus ``len(candidates)``).
        seed: Graine du tirage.

    Returns:
        La liste triée des indices tirés.
    """
    rng = random.Random(seed)
    return sorted(rng.sample(list(candidates), min(count, len(candidates))))


def select_flips(
    cosines: np.ndarray,
    twin_same_class: np.ndarray,
    copy_pred: np.ndarray,
    twin_pred: np.ndarray,
    count: int = 12,
    safe_cosine: float = SAFE_COSINE,
    seed: int = 42,
) -> list[int]:
    """Choisit des bascules (copie sûre dont la réponse diffère de celle de la jumelle).

    Args:
        cosines: Cosinus de chaque image PMRAM avec sa jumelle Kaggle.
        twin_same_class: Booléens « jumelle de même classe ».
        copy_pred: Classe prédite sur les copies.
        twin_pred: Classe prédite sur les jumelles.
        count: Nombre de bascules à tirer.
        safe_cosine: Cosinus minimal des copies sûres.
        seed: Graine du tirage.

    Returns:
        Indices (dans le jeu PMRAM) des bascules tirées.
    """
    flips = np.where(twin_same_class & (cosines >= safe_cosine) & (copy_pred != twin_pred))[0]
    return sample_indices(flips.tolist(), count, seed)


def select_pairs_by_bin(
    cosines: np.ndarray,
    labels: np.ndarray,
    bin_indices: Sequence[int],
    class_indices: Sequence[int],
    per_cell: int = 3,
    seed: int = 42,
) -> list[tuple[int, int, int]]:
    """Tire ``per_cell`` images par couple (tranche de cosinus, classe).

    Args:
        cosines: Cosinus de chaque image avec sa jumelle.
        labels: Classe (indice) de chaque image.
        bin_indices: Tranches à représenter (indices dans :data:`BIN_LABELS`).
        class_indices: Classes à représenter.
        per_cell: Nombre d'images par couple.
        seed: Graine du tirage.

    Returns:
        Liste de ``(indice de l'image, tranche, classe)``, tranche par tranche puis classe
        par classe.
    """
    bins = assign_bins(cosines)
    chosen: list[tuple[int, int, int]] = []
    for b in bin_indices:
        for c in class_indices:
            cell = np.where((bins == b) & (labels == c))[0].tolist()
            chosen += [(i, b, c) for i in sample_indices(cell, per_cell, seed)]
    return chosen
