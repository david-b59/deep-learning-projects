"""Tests de external_slices.py sur des volumes synthétiques (numpy)."""

import numpy as np
import pytest

from brain_mri.external_slices import (
    distance_to_mask_mm,
    same_grid,
    select_healthy_slices,
    slice_to_uint8,
    to_radiological,
    tumor_slice_index,
    window_bounds,
)


def make_volume(depth: int = 60) -> tuple[np.ndarray, np.ndarray]:
    """Fabrique un faux volume ``(20, 24, depth)`` (tête carrée) et son masque.

    Le masque couvre les coupes 28 à 31, avec la plus grande surface en 30.
    """
    volume = np.zeros((20, 24, depth), dtype=np.float32)
    volume[2:18, 2:22, 5 : depth - 5] = 100.0  # « tête » : coupes 5 à depth - 6
    mask = np.zeros(volume.shape, dtype=np.uint8)
    mask[8:10, 10:12, 28:32] = 1
    mask[7:12, 9:14, 30] = 1  # coupe 30 : surface maximale
    return volume, mask


def test_tumor_slice_is_largest_mask_slice() -> None:
    """La coupe tumeur est celle où le masque est le plus grand."""
    _, mask = make_volume()
    assert tumor_slice_index(mask) == 30


def test_tumor_slice_tie_takes_lowest_and_empty_raises() -> None:
    """Égalité : la plus basse ; masque vide : erreur."""
    mask = np.zeros((5, 5, 10), dtype=np.uint8)
    mask[1:3, 1:3, 4:7] = 1
    assert tumor_slice_index(mask) == 4
    with pytest.raises(ValueError):
        tumor_slice_index(np.zeros((5, 5, 10)))


def test_distance_uses_real_spacing() -> None:
    """La distance en mm = écart d'indices x espacement réel."""
    _, mask = make_volume()
    distance = distance_to_mask_mm(mask, 0.5)
    assert distance[30] == 0.0 and distance[28] == 0.0
    assert distance[20] == pytest.approx((28 - 20) * 0.5)
    assert distance[40] == pytest.approx((40 - 31) * 0.5)
    assert distance_to_mask_mm(mask, 2.0)[20] == pytest.approx(16.0)


def test_healthy_slices_are_far_and_have_brain() -> None:
    """Les coupes saines sont à 20 mm ou plus du masque et dans la tête."""
    volume, mask = make_volume()
    chosen = select_healthy_slices(mask, volume, spacing_z=0.5)  # 20 mm = 40 coupes
    assert chosen == [] or all(
        distance_to_mask_mm(mask, 0.5)[k] >= 20 for k in chosen
    )  # ici seule la fin du volume est assez loin
    chosen = select_healthy_slices(mask, volume, spacing_z=2.0)  # 20 mm = 10 coupes
    assert len(chosen) == 3
    distances = distance_to_mask_mm(mask, 2.0)
    assert all(distances[k] >= 20 and 32 <= k < 55 for k in chosen)  # seulement au-dessus
    assert chosen == sorted(chosen)


def test_healthy_slices_skip_patient_without_candidates() -> None:
    """Sans assez de coupes admissibles, le patient est écarté (liste vide)."""
    volume, mask = make_volume(depth=40)
    assert (
        select_healthy_slices(mask, volume, spacing_z=0.5) == []
    )  # 20 mm = 40 coupes : rien d admissible


def test_orientation_is_radiological() -> None:
    """Face en haut, droite du patient à gauche de l'image."""
    coupe = np.zeros((4, 6))  # indexée [x, y] ; x croît vers la droite, y vers l'avant
    coupe[3, 5] = 1.0  # voxel le plus à droite et le plus en avant
    image = to_radiological(coupe)
    assert image.shape == (6, 4)
    assert image[0, 0] == 1.0  # en haut (avant) et à gauche (droite du patient)


def test_slice_to_uint8_window_and_square() -> None:
    """Fenêtrage 8 bits et complétion en carré sans étirement."""
    volume, _ = make_volume()
    bounds = window_bounds(volume)
    assert bounds == (100.0, 100.0)
    coupe = np.linspace(0, 200, 20 * 24).reshape(20, 24)
    image = slice_to_uint8(coupe, (50.0, 150.0))
    assert image.dtype == np.uint8 and image.shape == (24, 24)
    assert image.min() == 0 and image.max() == 255


def test_same_grid() -> None:
    """Deux volumes sont recalés si dimensions et affine sont identiques."""
    affine = np.eye(4)
    assert same_grid(affine, (3, 3, 3), affine.copy(), (3, 3, 3))
    assert not same_grid(affine, (3, 3, 3), affine, (3, 3, 4))
    assert not same_grid(affine, (3, 3, 3), affine + 0.1, (3, 3, 3))


def test_load_volume_reorients_to_ras(tmp_path) -> None:
    """Un volume NIfTI en orientation LPS est réorienté en RAS (nibabel requis)."""
    nib = pytest.importorskip("nibabel")
    from brain_mri.external_slices import load_volume

    data = np.zeros((4, 5, 6), dtype=np.int16)
    data[0, 0, 0] = 7  # coin « gauche-postérieur-bas » de la grille d'origine
    affine = np.diag([-1.0, -1.0, 2.0, 1.0])  # x et y inversés : l'axe i va vers la gauche
    path = tmp_path / "v.nii.gz"
    nib.save(nib.Nifti1Image(data, affine), str(path))
    ras, zooms, _ = load_volume(path)
    assert ras.shape == (4, 5, 6) and zooms == (1.0, 1.0, 2.0)
    assert ras[3, 4, 0] == 7  # axes x et y retournés
