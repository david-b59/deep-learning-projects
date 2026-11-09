"""Tests des embeddings avec un petit DINOv2 à poids aléatoires (aucun téléchargement)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from transformers import Dinov2Config, Dinov2Model

from brain_mri.embeddings import embed_files, embed_images, preprocess


@pytest.fixture(scope="module")
def tiny_model() -> Dinov2Model:
    """Petit DINOv2 (32 dimensions, 2 couches) initialisé au hasard, en mode évaluation."""
    torch.manual_seed(0)
    config = Dinov2Config(
        hidden_size=32, num_hidden_layers=2, num_attention_heads=2, image_size=224, patch_size=14
    )
    return Dinov2Model(config).eval()


def make_scan(seed: int, size: int = 160) -> Image.Image:
    """Fabrique une fausse IRM : ellipse claire texturée sur fond noir."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size]
    inside = ((xx - size / 2) / (size / 3)) ** 2 + ((yy - size / 2) / (size / 2.5)) ** 2 <= 1
    array = np.zeros((size, size), dtype=np.uint8)
    array[inside] = rng.integers(80, 220, size=int(inside.sum()), dtype=np.uint8)
    return Image.fromarray(array)


def test_preprocess_shape_and_range() -> None:
    """Le prétraitement donne un tenseur (3, 224, 224) avec 3 canaux gris normalisés ImageNet."""
    tensor = preprocess(make_scan(0))
    assert tensor.shape == (3, 224, 224)
    assert tensor.dtype == torch.float32
    # Mêmes pixels sur les 3 canaux avant normalisation : on le retrouve en dé-normalisant
    back = tensor * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    back = back + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    assert torch.allclose(back[0], back[1], atol=1e-5)


def test_embed_images_unit_norm(tiny_model: Dinov2Model) -> None:
    """Chaque embedding a la bonne taille et une norme L2 égale à 1, quel que soit le lot."""
    images = [make_scan(i) for i in range(3)]
    vectors = embed_images(tiny_model, images, batch_size=2)
    assert vectors.shape == (3, 32)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)
    # Même résultat avec un autre découpage en lots
    again = embed_images(tiny_model, images, batch_size=3)
    assert np.allclose(vectors, again, atol=1e-5)


def test_embed_files_uses_cache(tiny_model: Dinov2Model, tmp_path: Path) -> None:
    """Le 2e appel relit le cache : mêmes vecteurs, aucun nouveau fichier de cache."""
    paths = []
    for i in range(3):
        path = tmp_path / f"scan_{i}.png"
        make_scan(i).save(path)
        paths.append(path)
    cache = tmp_path / "cache"
    first = embed_files(tiny_model, paths, cache, batch_size=2, workers=2)
    assert first.shape == (3, 32)
    assert len(list(cache.glob("*.npz"))) == 3
    calls = []
    second = embed_files(
        tiny_model, paths, cache, progress=lambda done, total: calls.append((done, total))
    )
    assert np.allclose(first, second)
    assert len(list(cache.glob("*.npz"))) == 3
    assert calls == [(3, 3)]  # tout vient du cache : aucun lot calculé
