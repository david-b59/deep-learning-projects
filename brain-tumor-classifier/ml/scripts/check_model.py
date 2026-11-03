"""Inspecte et vérifie le modèle avant tout entraînement.

Montre la fin du ResNet-50 pré-entraîné, la nouvelle tête et le nombre de paramètres
entraînables, puis passe un vrai lot dans le modèle et compare la loss de départ à
ln(nombre de classes), la loss d'un modèle qui répond au hasard.

Utilisation (depuis ``brain-tumor-classifier/ml/``) ::

    uv run python scripts/check_model.py
"""

from __future__ import annotations

import logging
import math

import torch
import truststore
from torch import nn
from torchvision import models
from torchvision.models import ResNet50_Weights

from brain_mri.config import load_config
from brain_mri.data import build_dataloaders
from brain_mri.model import build_model, count_parameters, unfreeze_last_block

# Les poids sont téléchargés une fois depuis download.pytorch.org : on utilise le
# magasin de certificats du système.
truststore.inject_into_ssl()


def fmt(number: int) -> str:
    """Formate un entier à la française, avec une espace comme séparateur de milliers."""
    return f"{number:,}".replace(",", " ")


def main() -> None:
    """Construit le modèle, affiche son contenu et fait une passe avant sur un lot."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config = load_config()
    bundle = build_dataloaders(config.data)
    num_classes = len(bundle.class_names)

    # Étape 1 : LIRE le modèle pré-entraîné avant de le modifier (comme dans le cours).
    original = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    print("Fin de print(model) :", *str(original).splitlines()[-3:], sep="\n  ")
    print("Dernière couche à remplacer :", original.fc)
    print(f"Paramètres d'origine        : {fmt(count_parameters(original)[1])}\n")

    # Étape 2 : construire notre modèle (corps gelé + nouvelle tête).
    torch.manual_seed(config.data.seed)  # même tête aléatoire à chaque exécution
    model = build_model(num_classes, config.model)

    print("Nouvelle tête :", model.fc)
    trainable, total = count_parameters(model)
    print(f"Paramètres    : {fmt(trainable)} entraînables / {fmt(total)} au total")
    print(
        "Couches entraînables :",
        sorted({n.split(".")[0] for n, p in model.named_parameters() if p.requires_grad}),
    )

    # Passe avant sur un lot : eval() et no_grad() car on ne fait que mesurer.
    images, labels = next(iter(bundle.train_loader))
    model.eval()
    with torch.no_grad():
        logits = model(images)
        loss = nn.CrossEntropyLoss()(logits, labels)
    print(f"\nLot en entrée : {tuple(images.shape)}")
    print(f"Logits        : {tuple(logits.shape)}  (un score par classe et par image)")
    chance = math.log(num_classes)
    print(f"Loss de départ: {loss.item():.3f}   (hasard = ln({num_classes}) = {chance:.3f})")

    unfreeze_last_block(model)
    trainable, _ = count_parameters(model)
    print(f"\nAprès dégel de layer4 (fine-tuning) : {fmt(trainable)} paramètres entraînables")


if __name__ == "__main__":
    main()
