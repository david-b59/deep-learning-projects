"""Extrait les caractéristiques avant la dernière couche des modèles A0, A1 et DINOv2 (11b-6a).

Pour A0 et A1 (ResNet-50) : la couche ``fc`` est remplacée par ``nn.Identity()``, ce qui donne
les 2048 caractéristiques juste avant la classification. Les trois parties (train, val, test)
du manifeste ``ml/splits/kaggle_grouped.csv`` sont passées avec la transformation d'ÉVALUATION
(sans augmentation). Résultat rangé dans ``data/cache/penultimate/<modèle>_<partie>.pt`` (hors
git) : jamais recalculé s'il existe ; écriture dans un fichier ``.tmp`` puis renommage.

Pour DINOv2 (B0 et B1 partagent le même corps) : on utilise ``get_features`` (vue d'évaluation,
cache ``data/cache/dinov2_features/``) ; seule la vue du test est à calculer.

Utilisation (depuis ``brain-tumor-classifier/ml/``) ::

    uv run python scripts/extract_penultimate.py --models a0 a1 dinov2
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from brain_mri.config import ModelConfig, load_config
from brain_mri.data import ManifestDataset, read_split_manifest
from brain_mri.dinov2_head import CacheMissingError, get_features
from brain_mri.embeddings import load_model
from brain_mri.model import build_model, get_device, load_checkpoint
from brain_mri.transforms import build_transforms

LOGGER = logging.getLogger("extract_penultimate")
ROOT = Path(__file__).resolve().parents[2]  # dossier brain-tumor-classifier/
CACHE_DIR = ROOT / "data" / "cache" / "penultimate"
PARTS = ("train", "val", "test")
RESNETS = {  # nom : (configuration, point de contrôle)
    "a0": (ROOT / "ml" / "configs" / "train_a0.yaml", ROOT / "models" / "a0" / "best.pt"),
    "a1": (ROOT / "ml" / "configs" / "train_a1.yaml", ROOT / "models" / "a1" / "best.pt"),
}
DINOV2_CONFIG = ROOT / "ml" / "configs" / "train_b0.yaml"


def extract_resnet(name: str) -> None:
    """Calcule et met en cache les caractéristiques 2048 d'un ResNet pour les trois parties.

    Args:
        name: ``a0`` ou ``a1``.
    """
    pending = [p for p in PARTS if not (CACHE_DIR / f"{name}_{p}.pt").is_file()]
    if not pending:
        LOGGER.info("%s : tout est déjà en cache", name)
        return
    config_path, checkpoint_path = RESNETS[name]
    config = load_config(config_path)
    samples, class_names = read_split_manifest(config.data)
    index = {c: i for i, c in enumerate(class_names)}
    device = get_device()
    checkpoint = load_checkpoint(checkpoint_path, device)
    model = build_model(len(class_names), ModelConfig(pretrained=False, freeze_backbone=False))
    model.load_state_dict(checkpoint["state_dict"])
    model.fc = nn.Identity()  # on lit les 2048 valeurs avant la classification
    model.to(device).eval()
    transform = build_transforms(config.data, train=False)  # évaluation : aucune augmentation
    for part in pending:
        items = [(path, index[label]) for path, label in samples[part]]
        loader = DataLoader(
            ManifestDataset(items, transform, class_names),
            batch_size=config.data.batch_size,
            shuffle=False,
            num_workers=0,
        )
        outputs = []
        start = time.time()
        with torch.no_grad():
            for images, _ in tqdm(loader, desc=f"{name} {part}", mininterval=60):
                outputs.append(model(images.to(device)).cpu())
        features = torch.cat(outputs)
        target = CACHE_DIR / f"{name}_{part}.pt"
        tmp = target.with_suffix(".tmp")
        torch.save({"features": features, "labels": torch.tensor([t for _, t in items])}, tmp)
        tmp.replace(target)  # renommage : jamais de cache à moitié écrit
        LOGGER.info("%s %s : %s en %.0f s", name, part, tuple(features.shape), time.time() - start)


def extract_dinov2() -> None:
    """Calcule (ou relit) les caractéristiques DINOv2 de la vue d'évaluation, trois parties."""
    config = load_config(DINOV2_CONFIG)
    samples, class_names = read_split_manifest(config.data)
    index = {c: i for i, c in enumerate(class_names)}
    backbone = None
    for part in PARTS:
        items = [(path, index[label]) for path, label in samples[part]]
        args = (part, items, class_names, config.data, config.dinov2, 1, False, config.data.seed)
        try:
            result = get_features(None, *args)
        except CacheMissingError:
            LOGGER.info("dinov2 %s : vue absente du cache, chargement du corps", part)
            if backbone is None:
                backbone = load_model(device=str(get_device()))
            result = get_features(backbone, *args, progress=LOGGER.info)
        LOGGER.info(
            "dinov2 %s : %s (%d vue(s) calculée(s))",
            part,
            tuple(result.features.shape),
            result.computed_views,
        )


def main() -> None:
    """Lit les options et lance les extractions l'une après l'autre."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--models",
        nargs="+",
        default=["a0", "a1", "dinov2"],
        choices=["a0", "a1", "dinov2"],
        help="Modèles à extraire",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for name in args.models:
        if name == "dinov2":
            extract_dinov2()
        else:
            extract_resnet(name)


if __name__ == "__main__":
    main()
