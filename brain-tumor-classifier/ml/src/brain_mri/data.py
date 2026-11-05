"""Pipeline de données : transformations, jeux de données, découpage stratifié et DataLoaders.

C'est la phase A du pipeline du cours (``ImageFolder`` + ``transforms`` +
``DataLoader``), écrite en fonctions réutilisables, configurables et testables.

Les trois jeux :
    - ``train`` : la majeure partie du dossier ``Training`` (stratifiée), avec une
      augmentation légère.
    - ``val`` : une part stratifiée de ce dossier (``val_fraction``), sans augmentation.
      Sert à régler l'entraînement et à l'arrêter au bon moment.
    - ``test`` : le dossier ``Testing``, sans augmentation. Utilisé une seule fois, à la fin.

Les images en double repérées par :mod:`brain_mri.duplicates` sont écartées d'abord :
une même image ne peut donc pas se trouver des deux côtés d'un découpage.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets
from torchvision.transforms import v2

from brain_mri.config import DataConfig
from brain_mri.duplicates import (
    IMAGE_SUFFIXES,
    TEST_DIR_NAME,
    TRAIN_DIR_NAME,
    load_or_create_report,
)
from brain_mri.transforms import build_transforms

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class DataBundle:
    """Tout ce dont l'entraînement et l'évaluation ont besoin côté données."""

    train_loader: DataLoader  # lots d'entraînement (mélangés)
    val_loader: DataLoader  # lots de validation
    test_loader: DataLoader  # lots de test
    class_names: list[str]  # noms des classes, triés (l'indice = l'étiquette)
    sizes: dict[str, int]  # nombre d'images de chaque jeu


def make_image_folder(
    root: Path, raw_dir: Path, excluded: set[str], transform: v2.Compose
) -> datasets.ImageFolder:
    """Crée un ``ImageFolder`` qui ignore les fichiers exclus (les doublons).

    Args:
        root: Dossier ``Training`` ou ``Testing`` (un sous-dossier par classe).
        raw_dir: Dossier par rapport auquel les chemins exclus sont exprimés.
        excluded: Chemins relatifs POSIX à ignorer.
        transform: Transformation appliquée à chaque image.

    Returns:
        Le jeu de données ; le nom du sous-dossier de chaque image sert d'étiquette.
    """

    def is_valid_file(path: str) -> bool:
        """Garde les fichiers image qui ne sont pas dans la liste d'exclusion."""
        file = Path(path)
        relative = file.relative_to(raw_dir).as_posix()
        return file.suffix.lower() in IMAGE_SUFFIXES and relative not in excluded

    return datasets.ImageFolder(root, transform=transform, is_valid_file=is_valid_file)


def stratified_split(
    targets: list[int], val_fraction: float, seed: int
) -> tuple[list[int], list[int]]:
    """Sépare des indices en entraînement et validation, en gardant les proportions de classes.

    Args:
        targets: Classe (indice) de chaque échantillon.
        val_fraction: Part des échantillons pour la validation, entre 0 et 1 (exclus).
        seed: Graine aléatoire (même graine = même découpage).

    Returns:
        ``(indices_entraînement, indices_validation)``, disjoints et triés.
    """
    train_idx, val_idx = train_test_split(
        np.arange(len(targets)),
        test_size=val_fraction,
        stratify=targets,  # chaque classe garde la même proportion dans les deux parties
        random_state=seed,
    )
    return sorted(train_idx.tolist()), sorted(val_idx.tolist())


def make_datasets(config: DataConfig) -> tuple[Dataset, Dataset, Dataset, list[str]]:
    """Construit les jeux d'entraînement, de validation et de test.

    Le dossier d'entraînement est chargé deux fois (avec et sans augmentation) et les
    mêmes indices stratifiés sont appliqués aux deux : la validation n'est donc jamais
    augmentée.

    Args:
        config: Configuration des données.

    Returns:
        ``(train, val, test, class_names)`` avec les noms de classes triés.

    Raises:
        ValueError: Si Training et Testing n'ont pas les mêmes classes.
    """
    report = load_or_create_report(config.raw_dir, config.processed_dir)
    excluded = report.files_to_exclude()
    LOGGER.info("%d fichier(s) en double écarté(s)", len(excluded))

    train_root = config.raw_dir / TRAIN_DIR_NAME
    test_root = config.raw_dir / TEST_DIR_NAME
    train_full = make_image_folder(
        train_root, config.raw_dir, excluded, build_transforms(config, train=True)
    )
    val_full = make_image_folder(
        train_root, config.raw_dir, excluded, build_transforms(config, train=False)
    )
    test = make_image_folder(
        test_root, config.raw_dir, excluded, build_transforms(config, train=False)
    )
    if train_full.classes != test.classes:
        raise ValueError(f"Classes différentes : {train_full.classes} et {test.classes}")

    train_idx, val_idx = stratified_split(train_full.targets, config.val_fraction, config.seed)
    return Subset(train_full, train_idx), Subset(val_full, val_idx), test, train_full.classes


def _seed_worker(worker_id: int) -> None:
    """Fixe la graine de Python et de NumPy dans chaque processus de chargement.

    Les augmentations ``v2`` utilisent le générateur aléatoire de torch, que PyTorch
    initialise lui-même pour chaque processus. Pour une augmentation entièrement
    reproductible, le script d'entraînement doit aussi appeler
    ``torch.manual_seed(config.seed)`` (prévu à l'étape 3).

    Args:
        worker_id: Numéro du processus de chargement (imposé par PyTorch, non utilisé).
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_dataloaders(config: DataConfig) -> DataBundle:
    """Construit les trois DataLoaders. Seul celui d'entraînement est mélangé.

    Args:
        config: Configuration des données.

    Returns:
        Un :class:`DataBundle` avec les DataLoaders, les noms de classes et les tailles.
    """
    train, val, test, class_names = make_datasets(config)
    generator = torch.Generator().manual_seed(config.seed)  # mélange reproductible

    def loader(dataset: Dataset, shuffle: bool) -> DataLoader:
        """Crée un DataLoader avec les réglages communs."""
        return DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=shuffle,  # True seulement pour l'entraînement
            num_workers=config.num_workers,
            worker_init_fn=_seed_worker,
            generator=generator if shuffle else None,
        )

    sizes = {"train": len(train), "val": len(val), "test": len(test)}
    LOGGER.info("Tailles des jeux : %s | classes : %s", sizes, class_names)
    return DataBundle(
        train_loader=loader(train, shuffle=True),
        val_loader=loader(val, shuffle=False),
        test_loader=loader(test, shuffle=False),
        class_names=class_names,
        sizes=sizes,
    )
