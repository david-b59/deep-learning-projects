"""Détection des quasi-doublons d'images par empreinte perceptuelle (dHash).

Pourquoi : un jeu externe ne sert de test honnête que si ses images ne sont PAS dans
le jeu d'entraînement. Le SHA-256 (voir ``duplicates.py``) ne voit que les copies
identiques à l'octet près ; une image redimensionnée, réenregistrée en JPEG ou
légèrement retouchée a un autre SHA-256 mais reste la même image. Le dHash la retrouve.

Principe du dHash (« difference hash ») : on passe l'image en niveaux de gris, on la
réduit à 9 colonnes × 8 lignes, puis, pour chaque ligne, on note 1 si un pixel est plus
clair que son voisin de gauche, 0 sinon. Cela donne 8 × 8 = 64 bits. Deux images
presque pareilles ont presque les mêmes bits ; la **distance de Hamming** compte les
bits qui diffèrent.

Seuil par défaut : 6 bits sur 64 (≈ 9 %). Deux images sans rapport ont en moyenne
~32 bits d'écart ; une copie recompressée ou redimensionnée en a 0 à 3. Le seuil 6
garde une marge pour de petites retouches. Limite constatée sur les IRM : les têtes
sombres sur fond noir se ressemblent à 9×8 pixels, donc au-delà de ~3 bits des faux
positifs apparaissent (images différentes). Le rapport garde la distance de chaque
correspondance pour pouvoir les trier. Le seuil se règle avec ``threshold``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from brain_mri.duplicates import IMAGE_SUFFIXES, sha256_file
from brain_mri.head_crop import crop_head

HASH_WIDTH = 9  # colonnes de la miniature : 9 pixels donnent 8 comparaisons par ligne
HASH_HEIGHT = 8  # lignes de la miniature : 8 lignes × 8 comparaisons = 64 bits
DEFAULT_THRESHOLD = 6  # distance de Hamming maximale (sur 64 bits) pour parler de quasi-doublon


@dataclass(frozen=True)
class Match:
    """Une image externe retrouvée dans le jeu de référence."""

    external: str  # chemin relatif de l'image externe
    reference: str  # chemin relatif de l'image de référence (Kaggle) qui lui ressemble
    distance: int  # distance de Hamming (0 pour un doublon exact par dHash)


def dhash(image: Image.Image) -> int:
    """Calcule l'empreinte dHash de 64 bits d'une image.

    Args:
        image: Image PIL, de n'importe quelle taille ou mode.

    Returns:
        Un entier de 64 bits (le bit de poids fort correspond au premier pixel).
    """
    gray = image.convert("L").resize((HASH_WIDTH, HASH_HEIGHT), Image.Resampling.LANCZOS)
    pixels = np.asarray(gray, dtype=np.int16)  # int16 : évite le dépassement sur les différences
    bits = (
        pixels[:, 1:] > pixels[:, :-1]
    ).ravel()  # vrai si le pixel est plus clair que son voisin de gauche
    value = 0
    for bit in bits:  # 64 bits assemblés dans un seul entier
        value = (value << 1) | int(bit)
    return value


def hamming(a: int, b: int) -> int:
    """Compte les bits qui diffèrent entre deux empreintes (distance de Hamming)."""
    return (a ^ b).bit_count()


def dhash_file(path: Path) -> int:
    """Calcule le dHash d'un fichier image."""
    with Image.open(path) as image:
        return dhash(image)


def list_images(root: Path) -> list[Path]:
    """Liste triée des images sous ``root`` (récursif).

    Raises:
        FileNotFoundError: Si ``root`` n'existe pas.
    """
    if not root.is_dir():
        raise FileNotFoundError(f"Dossier introuvable : {root}")
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def fingerprint_folders(roots: list[Path]) -> dict[str, tuple[str, int]]:
    """Calcule SHA-256 et dHash de toutes les images des dossiers donnés.

    Args:
        roots: Dossiers à parcourir (par ex. ``Training`` et ``Testing``).

    Returns:
        Un dictionnaire ``chemin relatif au parent du dossier -> (sha256, dhash)``.
        Le chemin commence par le nom du dossier (par ex. ``Training/glioma/x.jpg``).
    """
    result: dict[str, tuple[str, int]] = {}
    for root in roots:
        for path in list_images(root):
            relative = f"{root.name}/{path.relative_to(root).as_posix()}"
            result[relative] = (sha256_file(path), dhash_file(path))
    return result


def find_exact(
    external: dict[str, tuple[str, int]], reference: dict[str, tuple[str, int]]
) -> list[Match]:
    """Trouve les images externes dont le contenu est identique (même SHA-256) à une référence.

    Args:
        external: Empreintes du jeu externe (``chemin -> (sha256, dhash)``).
        reference: Empreintes du jeu de référence.

    Returns:
        Une :class:`Match` (distance 0) par image externe retrouvée.
    """
    by_sha: dict[str, str] = {}
    for path, (sha, _) in sorted(reference.items()):
        by_sha.setdefault(sha, path)  # première référence rencontrée
    return [
        Match(path, by_sha[sha], 0) for path, (sha, _) in sorted(external.items()) if sha in by_sha
    ]


def find_near(
    external: dict[str, tuple[str, int]],
    reference: dict[str, tuple[str, int]],
    threshold: int = DEFAULT_THRESHOLD,
) -> list[Match]:
    """Trouve, pour chaque image externe, la référence la plus proche si elle est assez proche.

    Args:
        external: Empreintes du jeu externe (``chemin -> (sha256, dhash)``).
        reference: Empreintes du jeu de référence.
        threshold: Distance de Hamming maximale (incluse) pour être un quasi-doublon.

    Returns:
        Une :class:`Match` par image externe dont la plus proche référence est à
        ``threshold`` bits ou moins. Vide si la référence est vide.
    """
    if not reference:
        return []
    ref_paths = sorted(reference)
    ref_hashes = np.array([reference[p][1] for p in ref_paths], dtype=np.uint64)
    matches = []
    for path, (_, value) in sorted(external.items()):
        # XOR puis comptage des bits à 1, pour toutes les références d'un coup
        distances = np.bitwise_count(ref_hashes ^ np.uint64(value))
        best = int(np.argmin(distances))
        if int(distances[best]) <= threshold:
            matches.append(Match(path, ref_paths[best], int(distances[best])))
    return matches


# --- Empreinte longue (256 bits) sur la tête recadrée, et voisins par embeddings ---------

LONG_HASH_SIZE = 16  # le dHash long compare 16 x 16 pixels voisins : 256 bits
# Seuils de décision sur le cosinus DINOv2 (embeddings de la tête recadrée). Calibration :
# ``scripts/calibrate_near_duplicates.py`` -> ``docs/figures/near_duplicate_calibration.json``.
# Justification : le seuil 0,94 retrouve environ 95 % des vraies copies connues (paires
# reconnues par le dHash 64 bits) tout en gardant ~2 % de fausses alertes sur des paires de
# classes différentes. On choisit une exclusion PRUDENTE : mieux vaut écarter une image à tort
# d'un jeu de test que laisser passer une copie de Kaggle.
DUPLICATE_COSINE = 0.94  # cosinus >= 0,94 avec n'importe quelle image Kaggle : copie, exclue
GRAY_ZONE_COSINE = 0.90  # [0,90 ; 0,94[ : zone grise, relue à l'œil sur une planche
CHUNK = 256  # nombre d'images traitées à la fois dans les calculs matriciels (limite la mémoire)


def dhash256(image: Image.Image) -> np.ndarray:
    """Calcule un dHash de 256 bits (16 x 16) sur la tête recadrée de l'image.

    L'image passe d'abord par :func:`brain_mri.head_crop.crop_head` : le fond noir et la
    taille de l'image n'influencent donc plus l'empreinte. Ensuite, même principe que
    :func:`dhash`, mais avec une miniature de 17 colonnes × 16 lignes.

    Args:
        image: Image PIL de n'importe quelle taille ou mode.

    Returns:
        Tableau uint8 de 32 octets (les 256 bits empaquetés).
    """
    head = crop_head(image)
    small = head.resize((LONG_HASH_SIZE + 1, LONG_HASH_SIZE), Image.Resampling.LANCZOS)
    pixels = np.asarray(small, dtype=np.int16)
    bits = pixels[:, 1:] > pixels[:, :-1]  # (16, 16) : plus clair que le voisin de gauche
    return np.packbits(bits.ravel())


def hamming_nearest(
    queries: np.ndarray, references: np.ndarray, exclude_self: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Trouve, pour chaque empreinte, la plus proche (distance de Hamming) parmi les références.

    Args:
        queries: Empreintes empaquetées, tableau uint8 ``(Q, octets)``.
        references: Empreintes de référence, tableau uint8 ``(R, octets)``.
        exclude_self: Si vrai, ``queries`` et ``references`` sont le même ensemble et une
            image ne peut pas être son propre voisin.

    Returns:
        ``(indices, distances)`` : deux tableaux de taille ``Q`` (indice de la référence la
        plus proche, et nombre de bits qui diffèrent).
    """
    indices = np.zeros(len(queries), dtype=np.int64)
    distances = np.zeros(len(queries), dtype=np.int64)
    for start in range(0, len(queries), CHUNK):
        block = queries[start : start + CHUNK]
        counts = np.bitwise_count(block[:, None, :] ^ references[None, :, :]).sum(axis=2)
        if exclude_self:
            counts[np.arange(len(block)), np.arange(start, start + len(block))] = 10**6
        indices[start : start + len(block)] = counts.argmin(axis=1)
        distances[start : start + len(block)] = counts.min(axis=1)
    return indices, distances


def nearest_cosine(
    queries: np.ndarray,
    references: np.ndarray,
    query_labels: np.ndarray | None = None,
    reference_labels: np.ndarray | None = None,
    other_class: bool = False,
    exclude_self: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Trouve, pour chaque embedding, le plus proche voisin par similarité cosinus.

    Les embeddings sont supposés de norme 1 : le produit scalaire est alors le cosinus.

    Args:
        queries: Embeddings ``(Q, dim)``.
        references: Embeddings de référence ``(R, dim)``.
        query_labels: Classe de chaque requête (utile avec ``other_class``).
        reference_labels: Classe de chaque référence.
        other_class: Si vrai, ne cherche que parmi les références d'une CLASSE DIFFÉRENTE
            (sert à fabriquer des paires « négatives » : même image impossible).
        exclude_self: Si vrai, ``queries`` et ``references`` sont le même ensemble et une
            image ne peut pas être son propre voisin.

    Returns:
        ``(indices, similarités)`` : deux tableaux de taille ``Q``.
    """
    indices = np.zeros(len(queries), dtype=np.int64)
    similarities = np.zeros(len(queries), dtype=np.float32)
    for start in range(0, len(queries), CHUNK):
        block = queries[start : start + CHUNK]
        sims = block @ references.T  # (bloc, R) : cosinus avec chaque référence
        if exclude_self:
            sims[np.arange(len(block)), np.arange(start, start + len(block))] = -np.inf
        if other_class:
            assert query_labels is not None and reference_labels is not None
            same = query_labels[start : start + len(block), None] == reference_labels[None, :]
            sims[same] = -np.inf
        indices[start : start + len(block)] = sims.argmax(axis=1)
        similarities[start : start + len(block)] = sims.max(axis=1)
    return indices, similarities


def similar_pairs(embeddings: np.ndarray, threshold: float) -> list[tuple[int, int, float]]:
    """Liste les paires d'images d'un même ensemble dont le cosinus atteint ``threshold``.

    Sert à repérer les doublons INTERNES d'un jeu (deux images du même jeu qui se
    ressemblent). Chaque paire n'apparaît qu'une fois, avec ``i < j``.

    Args:
        embeddings: Embeddings de norme 1, tableau ``(N, dim)``.
        threshold: Cosinus minimal (inclus) pour garder une paire.

    Returns:
        Liste de ``(i, j, cosinus)`` triée par ``i`` puis ``j``.
    """
    pairs: list[tuple[int, int, float]] = []
    for start in range(0, len(embeddings), CHUNK):
        block = embeddings[start : start + CHUNK]
        sims = block @ embeddings.T  # (bloc, N) : cosinus avec toutes les images
        rows, cols = np.nonzero(sims >= threshold)
        for row, col in zip(rows.tolist(), cols.tolist(), strict=True):
            if start + row < col:  # une seule fois par paire, jamais une image avec elle-même
                pairs.append((start + row, col, float(sims[row, col])))
    return pairs
