"""Tests du recadrage de la tête (``head_crop``) sur des images synthétiques."""

from __future__ import annotations

import numpy as np
from PIL import Image

from brain_mri.head_crop import crop_head, head_box, otsu_threshold


def make_scan(width: int = 300, height: int = 200) -> np.ndarray:
    """Fabrique une « tête » : ellipse claire décentrée sur fond noir, plus un artefact au coin."""
    yy, xx = np.mgrid[0:height, 0:width]
    image = np.zeros((height, width), dtype=np.uint8)
    inside = ((xx - 200) / 60.0) ** 2 + ((yy - 100) / 80.0) ** 2 <= 1.0  # centre (200, 100)
    image[inside] = 180
    image[5:8, 5:8] = 255  # petit artefact (3x3 pixels) dans le coin
    return image


def test_otsu_separates_two_levels() -> None:
    """Le seuil d'Otsu tombe entre les deux niveaux d'une image à deux valeurs."""
    values = np.concatenate([np.full(500, 20), np.full(500, 200)])
    assert 20 <= otsu_threshold(values) < 200


def test_head_box_ignores_small_artifact() -> None:
    """La boîte entoure l'ellipse (x 140-260, y 20-180) et ignore l'artefact du coin."""
    box = head_box(make_scan())
    assert box is not None
    left, top, right, bottom = box
    assert abs(left - 140) <= 4 and abs(right - 261) <= 4
    assert abs(top - 20) <= 4 and abs(bottom - 181) <= 4


def test_crop_head_is_square_and_centered() -> None:
    """La sortie est carrée, de mode L, et la tête y est centrée sans déformation."""
    result = crop_head(Image.fromarray(make_scan()))
    assert result.mode == "L"
    assert result.width == result.height
    array = np.asarray(result)
    ys, xs = np.nonzero(array > 128)
    assert abs(xs.mean() - result.width / 2) < 0.05 * result.width
    assert abs(ys.mean() - result.height / 2) < 0.05 * result.height
    # Ellipse de 120 de large sur 160 de haut : le rapport doit être conservé (pas d'étirement)
    ratio = (xs.max() - xs.min()) / (ys.max() - ys.min())
    assert abs(ratio - 120 / 160) < 0.05


def test_crop_head_accepts_rgb() -> None:
    """Une image RGB est acceptée et donne aussi une image carrée en niveaux de gris."""
    result = crop_head(Image.fromarray(make_scan()).convert("RGB"))
    assert result.mode == "L" and result.width == result.height


def test_crop_head_black_image_falls_back() -> None:
    """Une image toute noire ne plante pas : repli sur l'image entière, rendue carrée."""
    result = crop_head(Image.new("L", (120, 80), 0))
    assert result.width == result.height
    assert result.width >= 120
    assert not np.asarray(result).any()


def test_crop_head_keeps_brain_visible_with_bright_skull() -> None:
    """Crâne très blanc + cerveau gris : après étirement, le cerveau n'est pas écrasé.

    Avec un étirement fondé sur le masque d'Otsu (crâne seul), le cerveau gris (60 à 140)
    tomberait sous le plancher et deviendrait tout noir. Il doit garder son contraste.
    """
    yy, xx = np.mgrid[0:200, 0:200]
    radius = np.hypot(xx - 100, yy - 100)
    image = np.zeros((200, 200), dtype=np.uint8)
    image[radius <= 80] = 250  # crâne très blanc
    inner = radius <= 65
    image[inner] = (60 + (xx[inner] % 9) * 10).astype(np.uint8)  # cerveau : valeurs 60 à 140
    result = np.asarray(crop_head(Image.fromarray(image)))
    middle = result.shape[0] // 2
    centre = result[middle - 20 : middle + 20, middle - 20 : middle + 20].astype(float)
    assert centre.std() > 15  # le contraste du cerveau est conservé (pas écrasé)
    assert centre.mean() > 20  # et le cerveau n'est pas tout noir
    assert result.max() == 255  # le crâne reste au maximum


def test_crop_version_is_defined() -> None:
    """La version du prétraitement existe (elle sert de clé de cache des embeddings)."""
    from brain_mri.head_crop import CROP_VERSION

    assert isinstance(CROP_VERSION, str) and CROP_VERSION
