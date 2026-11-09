"""Audit du biais de source du jeu Kaggle (étape 11a).

Trois mesures, résumées en français et écrites dans ``docs/figures/audit_source_bias.json`` :

    1. tailles et modes des images par classe (Training et Testing) ;
    2. exactitude du modèle actuel (``models/best.pt``) sur le test, par classe selon que
       l'image fait 512x512 ou non ;
    3. exactitude d'un « classifieur idiot » qui ne voit que largeur, hauteur et mode.

Les tableaux 1 et 3 portent sur tous les fichiers des dossiers ; la mesure 2 porte sur
le jeu de test tel que le modèle le voit (doublons écartés), comme ``evaluate``.

Utilisation (depuis ``brain-tumor-classifier/ml/``) ::

    uv run python scripts/audit_source_bias.py
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch
from PIL import Image

from brain_mri.audit import (
    ImageInfo,
    accuracy_by_group,
    metadata_only_classifier,
    read_metadata,
    size_table,
)
from brain_mri.config import Config, ModelConfig, load_config
from brain_mri.data import build_dataloaders
from brain_mri.duplicates import TEST_DIR_NAME, TRAIN_DIR_NAME
from brain_mri.evaluate import predict
from brain_mri.model import build_model, get_device, load_checkpoint


def model_accuracy_by_group(config: Config) -> dict:
    """Mesure l'exactitude du meilleur modèle sur le test, par classe × taille.

    Args:
        config: Configuration complète.

    Returns:
        Le résultat de :func:`brain_mri.audit.accuracy_by_group` plus l'exactitude globale.

    Raises:
        FileNotFoundError: Si ``models/best.pt`` n'existe pas.
    """
    path = config.training.checkpoint_path
    if not path.is_file():
        raise FileNotFoundError(f"Modèle introuvable : {path}")
    device = get_device()
    bundle = build_dataloaders(config.data)
    checkpoint = load_checkpoint(path, device)  # weights_only=True géré par la fonction
    model = build_model(
        len(bundle.class_names), ModelConfig(pretrained=False, freeze_backbone=False)
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    y_true, y_pred = predict(model, bundle.test_loader, device)

    names = bundle.class_names
    samples = bundle.test_loader.dataset.samples  # (chemin, indice de classe) dans l'ordre
    infos = read_metadata_for([p for p, _ in samples])
    truth = [names[t] for t in y_true]
    pred = [names[p] for p in y_pred]
    return {
        "accuracy": sum(t == p for t, p in zip(truth, pred, strict=True)) / len(truth),
        "per_class": accuracy_by_group(infos, truth, pred),
    }


def read_metadata_for(paths: list[str]) -> list[ImageInfo]:
    """Lit les métadonnées d'une liste de fichiers, en gardant leur ordre.

    Args:
        paths: Chemins des images.

    Returns:
        Une :class:`ImageInfo` par chemin.
    """
    infos = []
    for p in paths:
        path = Path(p)
        with Image.open(path) as image:
            infos.append(ImageInfo(path, path.parent.name, *image.size, image.mode))
    return infos


def print_size_table(title: str, table: dict) -> None:
    """Affiche le tableau des tailles d'un dossier.

    Args:
        title: Titre (Training ou Testing).
        table: Résultat de :func:`brain_mri.audit.size_table`.
    """
    print(f"\n{title}")
    print(f"  {'classe':<11} {'images':>7} {'512x512':>9} {'tailles':>8} {'RGB':>7}")
    for label, row in table.items():
        print(
            f"  {label:<11} {row['n']:>7} {row['share_512']:>9.1%} "
            f"{row['n_distinct_sizes']:>8} {row['share_rgb']:>7.1%}"
        )


def main() -> None:
    """Lance l'audit, écrit le JSON et affiche le résumé en français."""
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    torch.manual_seed(0)
    config = load_config()
    raw = config.data.raw_dir
    train = read_metadata(raw / TRAIN_DIR_NAME)
    test = read_metadata(raw / TEST_DIR_NAME)

    tables = {"Training": size_table(train), "Testing": size_table(test)}
    model_result = model_accuracy_by_group(config)
    dumb = metadata_only_classifier(train, test)

    print_size_table("Training", tables["Training"])
    print_size_table("Testing", tables["Testing"])

    print(f"\nModèle actuel sur le test : exactitude {model_result['accuracy']:.2%}")
    print(f"  {'classe':<11} {'512x512':>18} {'autre taille':>18}")
    for label, groups in model_result["per_class"].items():
        cells = []
        for key in ("512x512", "autre"):
            g = groups[key]
            acc = "n/a" if g["accuracy"] is None else f"{g['accuracy']:.1%}"
            cells.append(f"{acc} ({g['correct']}/{g['n']})")
        print(f"  {label:<11} {cells[0]:>18} {cells[1]:>18}")

    print(
        f"\nClassifieur idiot (largeur, hauteur, mode seulement) : "
        f"exactitude {dumb['accuracy']:.2%} (hasard : {dumb['chance_level']:.2%})"
    )
    for label, row in dumb["per_class"].items():
        print(f"  {label:<11} {row['accuracy']:.1%} ({row['n']} images)")

    out = config.training.figures_dir / "audit_source_bias.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    report = {"size_tables": tables, "model_on_test": model_result, "metadata_only": dumb}
    with open(out, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)
    print(f"\nRapport enregistré dans {out}")


if __name__ == "__main__":
    main()
