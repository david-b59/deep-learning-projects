"""Tests de brain_mri.model (poids aléatoires : aucun téléchargement nécessaire)."""

from __future__ import annotations

import pytest
import torch

from brain_mri.config import ModelConfig
from brain_mri.model import build_model, count_parameters, unfreeze_last_block

NUM_CLASSES = 4
HEAD_PARAMS = 2048 * NUM_CLASSES + NUM_CLASSES  # poids + biais de la nouvelle couche Linear


@pytest.fixture(scope="module")
def frozen_model():
    """Fournit un ResNet-50 à poids aléatoires avec le corps gelé (construit une seule fois)."""
    return build_model(NUM_CLASSES, ModelConfig(pretrained=False, freeze_backbone=True))


def test_head_is_replaced_with_num_classes(frozen_model) -> None:
    """La nouvelle dernière couche relie 2048 caractéristiques au nombre de classes demandé."""
    assert frozen_model.fc.in_features == 2048
    assert frozen_model.fc.out_features == NUM_CLASSES


def test_only_the_head_is_trainable_when_frozen(frozen_model) -> None:
    """Avec le corps gelé, seule ``fc`` est entraînable (2048 * 4 + 4 paramètres)."""
    trainable_names = {
        n.split(".")[0] for n, p in frozen_model.named_parameters() if p.requires_grad
    }
    assert trainable_names == {"fc"}
    assert count_parameters(frozen_model)[0] == HEAD_PARAMS


def test_forward_gives_one_score_per_class(frozen_model) -> None:
    """Un lot de 3 images donne 3 lignes de NUM_CLASSES scores."""
    frozen_model.eval()
    with torch.no_grad():
        logits = frozen_model(torch.randn(3, 3, 64, 64))
    assert logits.shape == (3, NUM_CLASSES)


def test_everything_trainable_when_not_frozen() -> None:
    """Sans gel, tous les paramètres sont entraînables."""
    model = build_model(NUM_CLASSES, ModelConfig(pretrained=False, freeze_backbone=False))
    trainable, total = count_parameters(model)
    assert trainable == total


def test_unfreeze_last_block_makes_layer4_trainable() -> None:
    """Le fine-tuning dégèle ``layer4`` seulement, pas les blocs précédents."""
    model = build_model(NUM_CLASSES, ModelConfig(pretrained=False, freeze_backbone=True))
    unfreeze_last_block(model)
    assert all(p.requires_grad for p in model.layer4.parameters())
    assert not any(p.requires_grad for p in model.layer3.parameters())
