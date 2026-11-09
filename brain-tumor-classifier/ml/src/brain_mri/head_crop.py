"""Recadrage automatique de la tête sur une IRM, en carré, avec intensité normalisée.

Pourquoi : dans le jeu Kaggle, la tête n'occupe pas toujours la même part de l'image
(fonds noirs plus ou moins larges, images de tailles différentes). Pour comparer deux
IRM (embeddings, empreintes perceptuelles) on veut que seule la tête compte. Ce module
fournit :func:`crop_head`, qui :

    1. passe l'image en niveaux de gris et la floute légèrement ;
    2. cherche un seuil d'intensité (méthode d'Otsu) qui sépare la tête du fond ;
    3. garde les lignes et colonnes qui contiennent assez de pixels « tête » (robuste aux
       petits artefacts, au texte et aux bordures) pour en déduire la boîte englobante ;
    4. ajoute une petite marge puis des bandes noires pour obtenir un carré, sans déformer ;
    5. étire l'intensité entre les percentiles 1 et 99 mesurés sur les pixels de la boîte
       dont la valeur dépasse 10 % du maximum (le cerveau ET le crâne, même si le crâne
       est très blanc : le masque d'Otsu, lui, ne garderait que le crâne) ;
    6. renvoie une image 8 bits en niveaux de gris.

Si aucune tête n'est trouvée (image toute noire, ou uniforme), l'image entière est
utilisée, sans plantage.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter

DEFAULT_BLUR_RADIUS = 2.0  # rayon du flou gaussien (pixels) avant le seuillage
DEFAULT_MIN_FRACTION = 0.02  # part minimale de pixels « tête » dans une ligne/colonne retenue
DEFAULT_MARGIN = 0.03  # marge ajoutée autour de la boîte, en part de son plus grand côté
DEFAULT_PERCENTILES = (1.0, 99.0)  # bornes de l'étirement d'intensité, mesurées dans la tête
DEFAULT_FOREGROUND_FRACTION = 0.10  # un pixel « tête » dépasse cette part du maximum de la boîte
# Version du prétraitement : à changer à chaque modification qui change l'image rendue. Elle
# entre dans la clé du cache des embeddings, pour ne jamais relire d'anciens embeddings.
CROP_VERSION = "v2"  # v1 : percentiles sur le masque d'Otsu ; v2 : sur les pixels > 10 % du maximum


def otsu_threshold(values: np.ndarray) -> float:
    """Calcule le seuil d'Otsu d'un tableau de niveaux de gris 0-255.

    La méthode d'Otsu choisit le seuil qui sépare le mieux deux groupes de pixels
    (fond et tête) en maximisant la variance entre les groupes.

    Args:
        values: Tableau de valeurs entre 0 et 255 (n'importe quelle forme).

    Returns:
        Le seuil (un pixel est « tête » s'il est strictement au-dessus).
    """
    histogram = np.bincount(np.clip(values, 0, 255).astype(np.uint8).ravel(), minlength=256)
    histogram = histogram.astype(np.float64)
    total = histogram.sum()
    if total == 0:
        return 0.0
    levels = np.arange(256, dtype=np.float64)
    weight_bg = np.cumsum(histogram)  # nombre de pixels au niveau t ou en dessous
    weight_fg = total - weight_bg
    sum_bg = np.cumsum(histogram * levels)
    mean_bg = sum_bg / np.maximum(weight_bg, 1)
    mean_fg = (sum_bg[-1] - sum_bg) / np.maximum(weight_fg, 1)
    between = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2  # variance entre groupes
    return float(np.argmax(between))


def _span(counts: np.ndarray, min_count: float) -> tuple[int, int] | None:
    """Premier et dernier indices (inclus) où le compte atteint ``min_count``, ou None."""
    indices = np.flatnonzero(counts >= min_count)
    if indices.size == 0:
        return None
    return int(indices[0]), int(indices[-1])


def head_box(
    gray: np.ndarray,
    blur_radius: float = DEFAULT_BLUR_RADIUS,
    min_fraction: float = DEFAULT_MIN_FRACTION,
) -> tuple[int, int, int, int] | None:
    """Trouve la boîte englobante de la tête dans une image en niveaux de gris.

    Args:
        gray: Tableau 2D uint8 (hauteur, largeur).
        blur_radius: Rayon du flou gaussien appliqué avant le seuillage.
        min_fraction: Une ligne (colonne) compte si au moins cette part de ses pixels
            est au-dessus du seuil. Ignore ainsi les petits artefacts et le texte.

    Returns:
        ``(gauche, haut, droite, bas)`` avec droite et bas EXCLUS, ou ``None`` si aucune
        tête n'est détectée.
    """
    if gray.max() == gray.min():
        return None  # image uniforme (par ex. toute noire) : rien à détecter
    blurred = np.asarray(
        Image.fromarray(gray).filter(ImageFilter.GaussianBlur(blur_radius)), dtype=np.uint8
    )
    mask = blurred > otsu_threshold(blurred)
    if not mask.any():
        return None
    height, width = mask.shape
    rows = _span(mask.sum(axis=1), max(2.0, min_fraction * width))
    cols = _span(mask.sum(axis=0), max(2.0, min_fraction * height))
    if rows is None or cols is None:
        return None
    return cols[0], rows[0], cols[1] + 1, rows[1] + 1


def crop_head(
    image: Image.Image,
    blur_radius: float = DEFAULT_BLUR_RADIUS,
    min_fraction: float = DEFAULT_MIN_FRACTION,
    margin: float = DEFAULT_MARGIN,
    percentiles: tuple[float, float] = DEFAULT_PERCENTILES,
    foreground_fraction: float = DEFAULT_FOREGROUND_FRACTION,
) -> Image.Image:
    """Recadre la tête en carré et normalise son intensité.

    Args:
        image: Image PIL de n'importe quelle taille ou mode.
        blur_radius: Rayon du flou avant seuillage (pixels).
        min_fraction: Part minimale de pixels « tête » pour qu'une ligne/colonne compte.
        margin: Marge ajoutée autour de la boîte, en part de son plus grand côté.
        percentiles: Percentiles (bas, haut) de l'étirement d'intensité, calculés sur
            les pixels de la boîte plus clairs que ``foreground_fraction`` x le maximum.
        foreground_fraction: Part du maximum de la boîte au-dessus de laquelle un pixel
            compte pour l'étirement (0,10 = 10 %).

    Returns:
        Une image carrée 8 bits en niveaux de gris (mode ``L``). Si aucune tête n'est
        détectée, c'est l'image entière (sans recadrage), rendue carrée de la même façon.
    """
    gray = np.asarray(image.convert("L"), dtype=np.uint8)
    height, width = gray.shape
    box = head_box(gray, blur_radius, min_fraction)
    if box is None:
        box = (0, 0, width, height)  # repli : on garde toute l'image
    left, top, right, bottom = box

    # Marge puis carré : on étend le petit côté jusqu'au grand, centré, avec du noir
    side = max(right - left, bottom - top)
    side = int(round(side * (1 + 2 * margin)))
    x0 = int(round((left + right) / 2 - side / 2))
    y0 = int(round((top + bottom) / 2 - side / 2))
    canvas = np.zeros((side, side), dtype=np.uint8)  # bandes noires hors de l'image
    sx0, sy0 = max(x0, 0), max(y0, 0)
    sx1, sy1 = min(x0 + side, width), min(y0 + side, height)
    canvas[sy0 - y0 : sy1 - y0, sx0 - x0 : sx1 - x0] = gray[sy0:sy1, sx0:sx1]

    # Étirement d'intensité : percentiles mesurés sur les pixels de la boîte > 10 % du maximum.
    # (Otsu ne garderait que le crâne sur une image à crâne très blanc et écraserait le cerveau.)
    inside = canvas[canvas > foreground_fraction * canvas.max()]
    if inside.size < 10:
        inside = canvas.ravel()  # repli : toute l'image
    low, high = np.percentile(inside, percentiles)
    if high <= low:
        return Image.fromarray(canvas, mode="L")  # image uniforme : rien à étirer
    stretched = np.clip((canvas.astype(np.float32) - low) / (high - low), 0.0, 1.0)
    return Image.fromarray(np.round(stretched * 255).astype(np.uint8), mode="L")
