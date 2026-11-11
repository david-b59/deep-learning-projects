"""Extraction de coupes 2D depuis des IRM 3D externes (Pandit 2025), étape 11b-7.

But : fabriquer un test externe « intra-source » : pour un même patient (donc même hôpital,
même machine), des coupes AVEC tumeur et des coupes SANS tumeur. Cela casse le raccourci
« source = classe » du jeu Kaggle. TEST SEULEMENT, jamais d'entraînement.

Règles d'extraction (écrites avant toute mesure ; copie dans ``ml/splits/external_rules.md``) :

1. plan axial (axe z après réorientation en RAS) ;
2. coupe tumeur = coupe axiale où le masque a la plus grande surface (égalité : la plus basse) ;
3. coupes saines = coupes axiales du même patient situées AU-DESSUS de la tumeur et à au moins
   ``MIN_DISTANCE_MM`` = 20 mm de toute coupe contenant du masque (écart d'indices x espacement
   réel des voxels en z). Les coupes en dessous sont exclues : la planche de contrôle a montré
   qu'elles tombent sur le cou et la base du crâne, qui ne sont pas du cerveau ;
4. assez de cerveau : surface de tête (voxels > ``FOREGROUND_FRACTION`` = 15 % du percentile
   99,5 du volume) d'au moins ``MIN_BRAIN_FRACTION`` = 60 % de la surface maximale ;
5. ``HEALTHY_PER_PATIENT`` = 3 coupes saines aux quantiles 1/6, 3/6 et 5/6 des coupes admissibles
   (2 s'il n'y en a que 2 ; patient écarté s'il y en a moins de 2) ;
6. mêmes indices pour chaque séquence, seulement si sa grille est identique à celle de la T1 ;
7. orientation radiologique (comme Kaggle) : face en haut, droite du patient à gauche ;
8. intensité fenêtrée aux percentiles 0,5 et 99,5 des voxels non nuls du volume, 8 bits,
   complétée de noir en carré (aucun étirement).

Les fonctions de calcul travaillent sur des tableaux numpy déjà en orientation RAS
(axes : x vers la droite du patient, y vers l'avant, z vers le haut) ; seule
:func:`load_volume` dépend de nibabel.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

MIN_DISTANCE_MM = 20.0  # distance minimale entre une coupe saine et toute coupe masquée
FOREGROUND_FRACTION = 0.15  # un voxel « tête » dépasse cette part du percentile 99,5 du volume
MIN_BRAIN_FRACTION = 0.60  # surface de tête minimale, en part du maximum sur toutes les coupes
HEALTHY_PER_PATIENT = 3  # coupes saines visées par patient
MIN_HEALTHY_PER_PATIENT = 2  # en dessous, le patient est écarté
WINDOW_PERCENTILES = (0.5, 99.5)  # fenêtre d'intensité (voxels non nuls du volume)
TUMOR = "tumor"
HEALTHY = "healthy"


def load_volume(path: Path) -> tuple[np.ndarray, tuple[float, float, float], np.ndarray]:
    """Charge un volume NIfTI et le réoriente en RAS.

    Args:
        path: Fichier ``.nii`` ou ``.nii.gz``.

    Returns:
        ``(tableau 3D en RAS, espacement des voxels en mm (x, y, z), affine d'origine)``.
        Le tableau garde le type d'origine (pas de conversion en décimaux 64 bits).
    """
    import nibabel as nib  # import tardif : seule fonction qui en a besoin

    image = nib.load(str(path))
    canonical = nib.as_closest_canonical(image)  # axes (x, y, z) = (droite, avant, haut)
    data = np.asarray(canonical.dataobj)
    zooms = tuple(float(z) for z in canonical.header.get_zooms()[:3])
    return data, zooms, np.asarray(image.affine)


def same_grid(affine_a: np.ndarray, shape_a: tuple, affine_b: np.ndarray, shape_b: tuple) -> bool:
    """Dit si deux volumes sont recalés (même dimensions et même affine, à 1e-3 près).

    Args:
        affine_a: Affine du premier volume.
        shape_a: Dimensions du premier volume.
        affine_b: Affine du second volume.
        shape_b: Dimensions du second volume.

    Returns:
        True si les deux grilles sont identiques.
    """
    return tuple(shape_a) == tuple(shape_b) and bool(np.allclose(affine_a, affine_b, atol=1e-3))


def tumor_slice_index(mask: np.ndarray) -> int:
    """Donne l'indice de la coupe axiale où le masque est le plus grand.

    Args:
        mask: Masque 3D ``(x, y, z)`` (non nul = tumeur).

    Returns:
        L'indice z ; le plus bas en cas d'égalité.

    Raises:
        ValueError: Si le masque est vide.
    """
    areas = (mask > 0).sum(axis=(0, 1))
    if areas.max() == 0:
        raise ValueError("Masque vide : aucune coupe tumeur.")
    return int(np.argmax(areas))  # argmax renvoie la première (la plus basse) en cas d'égalité


def masked_slices(mask: np.ndarray) -> np.ndarray:
    """Liste les indices z des coupes qui contiennent du masque.

    Args:
        mask: Masque 3D.

    Returns:
        Tableau d'indices croissants.
    """
    return np.flatnonzero((mask > 0).sum(axis=(0, 1)) > 0)


def distance_to_mask_mm(mask: np.ndarray, spacing_z: float) -> np.ndarray:
    """Calcule, pour chaque coupe, la distance en mm à la coupe masquée la plus proche.

    Args:
        mask: Masque 3D.
        spacing_z: Espacement réel des voxels selon z, en mm.

    Returns:
        Tableau de longueur ``mask.shape[2]`` (0 sur les coupes contenant du masque).
    """
    tumor = masked_slices(mask)
    indices = np.arange(mask.shape[2])
    return np.abs(indices[:, None] - tumor[None, :]).min(axis=1) * spacing_z


def brain_slice_mask(volume: np.ndarray) -> np.ndarray:
    """Marque les coupes qui contiennent « assez de cerveau » (règle 4).

    Args:
        volume: Volume 3D d'intensités (T1).

    Returns:
        Tableau de booléens de longueur ``volume.shape[2]``.
    """
    threshold = FOREGROUND_FRACTION * np.percentile(volume[volume > 0], WINDOW_PERCENTILES[1])
    areas = (volume > threshold).sum(axis=(0, 1))
    return areas >= MIN_BRAIN_FRACTION * areas.max()


def select_healthy_slices(
    mask: np.ndarray, volume: np.ndarray, spacing_z: float, count: int = HEALTHY_PER_PATIENT
) -> list[int]:
    """Choisit les coupes saines d'un patient, au-dessus de la tumeur (règles 3, 4 et 5).

    Args:
        mask: Masque 3D de la tumeur.
        volume: Volume 3D T1 (pour juger la quantité de cerveau).
        spacing_z: Espacement des voxels selon z, en mm.
        count: Nombre de coupes voulu.

    Returns:
        Indices z choisis (aux quantiles 1/(2n), 3/(2n), ... des coupes admissibles), liste
        vide si le patient a moins de ``MIN_HEALTHY_PER_PATIENT`` coupes admissibles.
    """
    far = distance_to_mask_mm(mask, spacing_z) >= MIN_DISTANCE_MM
    above = np.arange(mask.shape[2]) > masked_slices(mask).max()  # au-dessus de la tumeur
    candidates = np.flatnonzero(far & above & brain_slice_mask(volume))
    if len(candidates) < MIN_HEALTHY_PER_PATIENT:
        return []
    n = min(count, len(candidates))
    positions = [int((2 * j + 1) * len(candidates) / (2 * n)) for j in range(n)]
    return [int(candidates[p]) for p in positions]


def to_radiological(slice_xy: np.ndarray) -> np.ndarray:
    """Met une coupe RAS ``(x, y)`` en orientation radiologique (comme Kaggle).

    En RAS, x croît vers la droite du patient et y vers l'avant. Image radiologique : face en
    haut (lignes = y décroissant) et droite du patient à gauche (colonnes = x décroissant).

    Args:
        slice_xy: Coupe 2D indexée ``[x, y]``.

    Returns:
        Coupe 2D indexée ``[ligne, colonne]``.
    """
    return slice_xy.T[::-1, ::-1]


def window_bounds(volume: np.ndarray) -> tuple[float, float]:
    """Calcule la fenêtre d'intensité d'un volume (percentiles 0,5 et 99,5 des voxels non nuls).

    Args:
        volume: Volume 3D.

    Returns:
        ``(bas, haut)``.
    """
    low, high = np.percentile(volume[volume > 0], WINDOW_PERCENTILES)
    return float(low), float(high)


def slice_to_uint8(slice_xy: np.ndarray, bounds: tuple[float, float]) -> np.ndarray:
    """Convertit une coupe en image 8 bits radiologique, carrée (complétée de noir).

    Args:
        slice_xy: Coupe 2D indexée ``[x, y]`` (orientation RAS).
        bounds: Fenêtre ``(bas, haut)`` de :func:`window_bounds`.

    Returns:
        Tableau ``uint8`` carré, niveaux de gris.
    """
    low, high = bounds
    scaled = np.clip((slice_xy.astype(np.float32) - low) / max(high - low, 1e-6), 0.0, 1.0)
    image = np.rint(to_radiological(scaled) * 255).astype(np.uint8)
    side = max(image.shape)
    pad_rows, pad_cols = side - image.shape[0], side - image.shape[1]
    return np.pad(
        image,
        ((pad_rows // 2, pad_rows - pad_rows // 2), (pad_cols // 2, pad_cols - pad_cols // 2)),
    )


def save_png(image: np.ndarray, path: Path) -> None:
    """Enregistre une image 8 bits en PNG (dossier créé au besoin).

    Args:
        image: Tableau ``uint8`` 2D.
        path: Fichier de sortie.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(path)
