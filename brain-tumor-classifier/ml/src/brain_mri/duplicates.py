"""Détection des images en double, dans et entre les dossiers Training et Testing.

Pourquoi c'est important : si la même image est dans Training et dans Testing, le
modèle est « testé » sur une image qu'il a déjà vue, et le score de test est gonflé.
Deux fichiers sont considérés comme doublons quand leurs octets sont identiques
(même SHA-256). Les quasi-doublons (même scanner réenregistré, retourné ou
redimensionné) ne sont PAS détectés : c'est une limite connue de cette vérification.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

LOGGER = logging.getLogger(__name__)

IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png"})  # extensions d'images acceptées
TRAIN_DIR_NAME = "Training"  # nom du dossier d'entraînement dans le jeu de données
TEST_DIR_NAME = "Testing"  # nom du dossier de test dans le jeu de données
REPORT_NAME = "duplicates_report.json"  # nom du rapport enregistré


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """Calcule l'empreinte SHA-256 d'un fichier, lu par morceaux (peu de mémoire).

    Args:
        path: Fichier à lire.
        chunk_size: Taille des morceaux lus, en octets (1 Mo par défaut).

    Returns:
        L'empreinte en hexadécimal : deux fichiers identiques donnent la même.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        while chunk := file.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def hash_images(root: Path) -> dict[str, list[str]]:
    """Calcule l'empreinte de chaque image sous ``root``.

    Args:
        root: Dossier à parcourir récursivement.

    Returns:
        Un dictionnaire ``empreinte -> [chemins relatifs POSIX]``, trié pour être
        déterministe.

    Raises:
        FileNotFoundError: Si ``root`` n'existe pas.
    """
    if not root.is_dir():
        raise FileNotFoundError(f"Dossier introuvable : {root}")
    groups: dict[str, list[str]] = defaultdict(list)
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            groups[sha256_file(path)].append(path.relative_to(root).as_posix())
    return dict(groups)


@dataclass
class DuplicateReport:
    """Résultat de la vérification des doublons. Les chemins sont relatifs au dossier brut."""

    n_train: int = 0  # nombre d'images dans Training
    n_test: int = 0  # nombre d'images dans Testing
    # Fichiers de test dont le contenu existe aussi dans Training (fuite de données).
    test_files_in_train: list[str] = field(default_factory=list)
    # Groupes de fichiers identiques dans Training (peuvent fuiter vers la validation).
    train_groups: list[list[str]] = field(default_factory=list)
    # Groupes de fichiers identiques dans Testing.
    test_groups: list[list[str]] = field(default_factory=list)

    def files_to_exclude(self) -> set[str]:
        """Donne les fichiers que le pipeline de données ne doit pas utiliser.

        - Les fichiers de test qui existent aussi dans Training (fuite vers le test).
        - Les copies en trop dans un groupe de fichiers identiques : le premier chemin
          (par ordre alphabétique) est gardé, les autres sont écartés. Ainsi, les
          copies d'une même image ne se retrouvent pas des deux côtés du découpage
          entraînement / validation.
        """
        excluded = set(self.test_files_in_train)
        for group in (*self.train_groups, *self.test_groups):
            excluded.update(sorted(group)[1:])  # [1:] : tout sauf le premier
        return excluded

    def to_json(self) -> str:
        """Sérialise le rapport en texte JSON lisible."""
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, text: str) -> DuplicateReport:
        """Reconstruit un rapport à partir du texte produit par :meth:`to_json`."""
        return cls(**json.loads(text))


def find_duplicates(raw_dir: Path) -> DuplicateReport:
    """Compare par contenu les images de ``Training`` et de ``Testing``.

    Args:
        raw_dir: Dossier qui contient ``Training/`` et ``Testing/``.

    Returns:
        Le rapport de doublons.
    """
    train = hash_images(raw_dir / TRAIN_DIR_NAME)
    test = hash_images(raw_dir / TEST_DIR_NAME)

    def prefixed(split: str, paths: list[str]) -> list[str]:
        """Ajoute le nom du dossier (Training ou Testing) devant chaque chemin relatif."""
        return [f"{split}/{p}" for p in paths]

    # Une empreinte présente des deux côtés = image de test déjà vue à l'entraînement.
    leaked = sorted(
        path
        for digest, paths in test.items()
        if digest in train
        for path in prefixed(TEST_DIR_NAME, paths)
    )
    report = DuplicateReport(
        n_train=sum(len(p) for p in train.values()),
        n_test=sum(len(p) for p in test.values()),
        test_files_in_train=leaked,
        # Plus d'un chemin pour une même empreinte = groupe d'images identiques.
        train_groups=[prefixed(TRAIN_DIR_NAME, p) for p in train.values() if len(p) > 1],
        test_groups=[prefixed(TEST_DIR_NAME, p) for p in test.values() if len(p) > 1],
    )
    LOGGER.info(
        "Doublons : %d image(s) de test aussi dans Training, %d groupe(s) dans Training, "
        "%d groupe(s) dans Testing",
        len(report.test_files_in_train),
        len(report.train_groups),
        len(report.test_groups),
    )
    return report


def save_report(report: DuplicateReport, processed_dir: Path) -> Path:
    """Écrit le rapport en JSON dans ``processed_dir`` et renvoie son chemin."""
    processed_dir.mkdir(parents=True, exist_ok=True)
    path = processed_dir / REPORT_NAME
    path.write_text(report.to_json(), encoding="utf-8")
    return path


def count_images(root: Path) -> int:
    """Compte les fichiers image sous ``root`` (sans calcul d'empreinte, donc rapide)."""
    return sum(1 for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def load_or_create_report(raw_dir: Path, processed_dir: Path) -> DuplicateReport:
    """Charge le rapport en cache, ou le calcule puis l'enregistre.

    Le cache est recalculé s'il n'existe pas, ou si ses nombres d'images ne
    correspondent plus aux fichiers présents sur le disque (données modifiées ou
    retéléchargées).

    Args:
        raw_dir: Dossier qui contient ``Training/`` et ``Testing/``.
        processed_dir: Dossier où le rapport est enregistré.

    Returns:
        Le rapport de doublons à jour.
    """
    path = processed_dir / REPORT_NAME
    if path.is_file():
        cached = DuplicateReport.from_json(path.read_text(encoding="utf-8"))
        if cached.n_train == count_images(
            raw_dir / TRAIN_DIR_NAME
        ) and cached.n_test == count_images(raw_dir / TEST_DIR_NAME):
            return cached
        LOGGER.warning("Le rapport de doublons en cache est périmé : recalcul")
    report = find_duplicates(raw_dir)
    save_report(report, processed_dir)
    return report
