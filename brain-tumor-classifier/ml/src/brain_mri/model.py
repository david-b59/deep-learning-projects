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

from pathlib import Path

import torch
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


def build_model(num_classes: int, config: ModelConfig | None = None) -> nn.Module:
    """Crée le modèle choisi par ``config.architecture`` (ResNet-50 par défaut).

    Avec ``architecture = "dinov2_s"``, renvoie un
    :class:`brain_mri.dinov2_head.Dinov2Classifier` (DINOv2-small gelé + couche linéaire) ;
    les autres réglages (``pretrained``, ``freeze_backbone``) ne servent alors pas.

    Args:
        num_classes: Nombre de classes en sortie (4 ici).
        config: Réglages du modèle (poids pré-entraînés, gel du corps). Par défaut,
            ``ModelConfig()``.

    Returns:
        Le modèle. La nouvelle tête ``fc`` est toujours entraînable ; le reste est
        gelé si ``config.freeze_backbone`` vaut True.
    """
    config = config or ModelConfig()
    if config.architecture == "dinov2_s":
        # Import ici : le ResNet n'a pas besoin de charger la bibliothèque transformers.
        from brain_mri.dinov2_head import Dinov2Classifier

        return Dinov2Classifier(num_classes)
    weights = ResNet50_Weights.IMAGENET1K_V1 if config.pretrained else None
    model = models.resnet50(weights=weights)
    if config.freeze_backbone:
        freeze(model)
    # Important : on remplace fc APRÈS le gel. La nouvelle couche est créée avec
    # requires_grad = True, donc seule elle sera entraînée.
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def build_model_for_checkpoint(checkpoint: dict, config: ModelConfig | None = None) -> nn.Module:
    """Crée le modèle vide qui correspond à un checkpoint, prêt pour ``load_state_dict``.

    L'architecture est lue dans le checkpoint (clé ``architecture``) ; les anciens checkpoints
    du ResNet n'ont pas cette clé : on prend alors celle de ``config`` (``resnet50`` par défaut).

    Args:
        checkpoint: Dictionnaire lu par :func:`load_checkpoint`.
        config: Configuration du modèle, utilisée si le checkpoint n'indique pas l'architecture.

    Returns:
        Le modèle, sans poids pré-entraînés téléchargés (sauf le corps DINOv2, épinglé),
        en attente des poids du checkpoint.
    """
    architecture = checkpoint.get("architecture") or (config or ModelConfig()).architecture
    model_config = ModelConfig(architecture=architecture, pretrained=False, freeze_backbone=False)
    return build_model(len(checkpoint["class_names"]), model_config)


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


def get_device() -> torch.device:
    """Choisit le GPU s'il est disponible, sinon le CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_checkpoint(path: Path, device: torch.device | str = "cpu") -> dict:
    """Relit un fichier produit par :func:`save_checkpoint`.

    Args:
        path: Fichier à lire.
        device: Appareil sur lequel charger les tenseurs.

    Returns:
        Le dictionnaire enregistré (``state_dict``, ``class_names``, etc.).
    """
    return torch.load(path, map_location=device, weights_only=True)
