"""Tests du module ``near_duplicates`` sur de fausses images (aucune donnée réelle)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance

from brain_mri.near_duplicates import (
    dhash,
    dhash256,
    dhash_file,
    find_exact,
    find_near,
    fingerprint_folders,
    hamming,
    hamming_nearest,
    nearest_cosine,
    similar_pairs,
)


def make_image(seed: int, size: int = 128) -> Image.Image:
    """Fabrique une image « douce » reproductible (bruit 16×16 agrandi), en niveaux de gris."""
    rng = np.random.default_rng(seed)
    small = rng.integers(0, 256, size=(16, 16), dtype=np.uint8)
    return Image.fromarray(small).resize((size, size), Image.Resampling.BICUBIC)


def test_hamming_counts_differing_bits() -> None:
    """La distance de Hamming compte exactement les bits différents."""
    assert hamming(0b1010, 0b1010) == 0
    assert hamming(0b1010, 0b0101) == 4
    assert hamming(0, 1) == 1


def test_dhash_fits_in_64_bits_and_is_deterministic() -> None:
    """Le dHash tient sur 64 bits et donne toujours la même valeur pour la même image."""
    image = make_image(1)
    assert 0 <= dhash(image) < 2**64
    assert dhash(image) == dhash(image.copy())


def test_identical_image_has_distance_zero(tmp_path: Path) -> None:
    """Une image enregistrée deux fois a la même empreinte (distance 0)."""
    make_image(1).save(tmp_path / "a.png")
    make_image(1).save(tmp_path / "b.png")
    assert hamming(dhash_file(tmp_path / "a.png"), dhash_file(tmp_path / "b.png")) == 0


def test_resized_and_slightly_modified_image_is_near_duplicate() -> None:
    """Une version redimensionnée et un peu plus lumineuse reste très proche."""
    original = make_image(1, size=128)
    resized = original.resize((96, 96), Image.Resampling.BILINEAR)
    brighter = ImageEnhance.Brightness(original).enhance(1.1)
    assert hamming(dhash(original), dhash(resized)) <= 6
    assert hamming(dhash(original), dhash(brighter)) <= 6


def test_different_image_is_far() -> None:
    """Deux images sans rapport sont bien au-dessus du seuil de 6 bits."""
    assert hamming(dhash(make_image(1)), dhash(make_image(2))) > 6


def test_find_near_and_exact_on_folders(tmp_path: Path) -> None:
    """Sur des dossiers : copie exacte, version redimensionnée et image différente."""
    reference_dir = tmp_path / "Training" / "glioma"
    external_dir = tmp_path / "ext" / "glioma"
    reference_dir.mkdir(parents=True)
    external_dir.mkdir(parents=True)
    make_image(1).save(reference_dir / "r1.png")
    make_image(2).save(reference_dir / "r2.png")
    make_image(1).save(external_dir / "exact.png")  # octets identiques à r1.png
    make_image(2).resize((96, 96)).save(external_dir / "near.png")  # autre taille, même image
    make_image(3).save(external_dir / "other.png")  # image différente

    reference = fingerprint_folders([tmp_path / "Training"])
    external = fingerprint_folders([tmp_path / "ext"])

    exact = find_exact(external, reference)
    assert [m.external for m in exact] == ["ext/glioma/exact.png"]
    assert exact[0].reference == "Training/glioma/r1.png"

    near = {m.external: m for m in find_near(external, reference, threshold=6)}
    assert set(near) == {"ext/glioma/exact.png", "ext/glioma/near.png"}
    assert near["ext/glioma/near.png"].reference == "Training/glioma/r2.png"
    assert "ext/glioma/other.png" not in near


def test_find_near_with_empty_reference() -> None:
    """Sans référence, il n'y a aucun quasi-doublon (et pas d'erreur)."""
    assert find_near({"x.png": ("abc", 5)}, {}) == []


def test_dhash256_is_32_bytes_and_stable_to_border() -> None:
    """Le dHash long fait 32 octets et ne change pas quand on ajoute un fond noir autour."""
    yy, xx = np.mgrid[0:120, 0:120]
    rng = np.random.default_rng(3)
    head = np.zeros((120, 120), dtype=np.uint8)
    inside = (xx - 60) ** 2 + (yy - 60) ** 2 <= 50**2
    head[inside] = rng.integers(60, 220, size=int(inside.sum()), dtype=np.uint8)
    padded = np.zeros((200, 260), dtype=np.uint8)
    padded[30:150, 70:190] = head  # même tête, fond noir plus grand, décalée
    a, b = dhash256(Image.fromarray(head)), dhash256(Image.fromarray(padded))
    assert a.shape == (32,) and a.dtype == np.uint8
    assert np.bitwise_count(a ^ b).sum() <= 10  # presque identiques sur 256 bits


def test_hamming_nearest_finds_closest_and_excludes_self() -> None:
    """Le voisin le plus proche est trouvé ; avec exclude_self, l'image ne se choisit pas."""
    rng = np.random.default_rng(0)
    refs = rng.integers(0, 256, size=(5, 32), dtype=np.uint8)
    query = refs[2:3].copy()
    query[0, 0] ^= 0b11  # 2 bits de différence
    indices, distances = hamming_nearest(query, refs)
    assert indices[0] == 2 and distances[0] == 2
    indices, distances = hamming_nearest(refs, refs, exclude_self=True)
    assert all(indices != np.arange(5)) and all(distances > 0)


def test_nearest_cosine_other_class_and_self() -> None:
    """Le voisin cosinus respecte les options « autre classe » et « sans soi-même »."""
    vectors = np.array([[1, 0], [0.99, 0.141], [0, 1], [0.1, 0.995]], dtype=np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    labels = np.array(["a", "a", "b", "b"])
    idx, sim = nearest_cosine(vectors, vectors, exclude_self=True)
    assert list(idx) == [1, 0, 3, 2] and (sim > 0.9).all()
    idx, sim = nearest_cosine(vectors, vectors, labels, labels, other_class=True)
    assert idx[0] == 3 and idx[2] == 1  # vecteur 0 -> vecteur 3 (classe b), 2 -> 1 (classe a)


def test_similar_pairs_lists_each_pair_once() -> None:
    """Seules les paires au-dessus du seuil sont listées, chacune une seule fois (i < j)."""
    vectors = np.array([[1, 0], [0.99, 0.141], [0, 1], [0.1, 0.995]], dtype=np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    pairs = similar_pairs(vectors, 0.9)
    assert [(i, j) for i, j, _ in pairs] == [(0, 1), (2, 3)]
    assert all(c >= 0.9 for _, _, c in pairs)
    assert similar_pairs(vectors, 0.9999) == []
