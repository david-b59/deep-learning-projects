"""Modèle : un ResNet-50 pré-entraîné avec une nouvelle tête à 4 classes (transfer learning).

C'est la phase B du pipeline du cours. La recette :
    1. charger ResNet-50 avec les poids ImageNet ;
    2. geler tous les paramètres pré-entraînés (``requires_grad = False``) ;
    3. remplacer la dernière couche ``fc`` (2048 -> 1000 classes ImageNet) par une
       nouvelle ``Linear(2048, num_classes)``, entraînable par défaut ;
    4. plus tard (fine-tuning), dégeler le dernier bloc ``layer4``.

Le nom ``fc`` a été lu dans ``print(model)``, il n'a pas été supposé.
"""

from __future__ import annotations

from torch import nn
from torchvision import models
from torchvision.models import ResNet50_Weights

from brain_mri.config import ModelConfig


def freeze(module: nn.Module) -> None:
    """Fige tous les paramètres de ``module`` : ils ne seront plus mis à jour."""
    for param in module.parameters():
        param.requires_grad = False


def unfreeze(module: nn.Module) -> None:
    """Rend tous les paramètres de ``module`` entraînables."""
    for param in module.parameters():
        param.requires_grad = True


def build_model(num_classes: int, config: ModelConfig | None = None) -> models.ResNet:
    """Crée un ResNet-50 avec une nouvelle tête de classification.

    Args:
        num_classes: Nombre de classes en sortie (4 ici).
        config: Réglages du modèle (poids pré-entraînés, gel du corps). Par défaut,
            ``ModelConfig()``.

    Returns:
        Le modèle. La nouvelle tête ``fc`` est toujours entraînable ; le reste est
        gelé si ``config.freeze_backbone`` vaut True.
    """
    config = config or ModelConfig()
    weights = ResNet50_Weights.IMAGENET1K_V1 if config.pretrained else None
    model = models.resnet50(weights=weights)
    if config.freeze_backbone:
        freeze(model)
    # Important : on remplace fc APRÈS le gel. La nouvelle couche est créée avec
    # requires_grad = True, donc seule elle sera entraînée.
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def unfreeze_last_block(model: models.ResNet) -> None:
    """Étape de fine-tuning : rend le dernier bloc résiduel ``layer4`` entraînable."""
    unfreeze(model.layer4)


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Compte les paramètres d'un modèle.

    Args:
        model: Le modèle à analyser.

    Returns:
        ``(entraînables, total)``.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total
