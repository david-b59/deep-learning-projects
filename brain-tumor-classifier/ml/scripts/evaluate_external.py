"""Mesure du modèle actuel (``models/best.pt``) sur le jeu externe PMRAM (étape 11a).

Trois sous-ensembles (voir :mod:`brain_mri.external_eval`) : strict, seuil 2, doublons.
Prétraitement identique au test Kaggle. Résultat dans ``docs/figures/external_pmram.json``.

Point d'attention : toutes les images PMRAM sont en 512x512 RGB, y compris « notumor »,
alors qu'à l'entraînement « notumor » n'était presque jamais en 512x512. C'est donc un
test du raccourci de taille dans le sens inverse : si le modèle lit la taille, il prendra
les « notumor » PMRAM pour des tumeurs.

Utilisation (depuis ``brain-tumor-classifier/ml/``) ::

    uv run python scripts/evaluate_external.py
"""

from __future__ import annotations

import json
import logging

import torch

from brain_mri.config import ModelConfig, load_config
from brain_mri.external_eval import SUBSET_NAMES, evaluate_subset, select_subsets
from brain_mri.model import build_model, get_device, load_checkpoint
from brain_mri.transforms import build_transforms

TITLES = {
    "strict": "A. Strict (statut kept, seuil 6 bits)",
    "seuil_2": "B. Seuil 2 (pas de jumelle à 2 bits ou moins)",
    "doublons": "C. Doublons (jumelle à 2 bits ou moins, contrôle)",
}


def print_subset(title: str, res: dict, class_names: list[str]) -> None:
    """Affiche en français le résumé d'un sous-ensemble.

    Args:
        title: Titre du sous-ensemble.
        res: Résultat de ``evaluate_subset``.
        class_names: Noms des classes.
    """
    print(f"\n{title} : {res['n']} images")
    if res["n"] == 0:
        return
    print(f"  exactitude : {res['accuracy']:.2%}")
    for name in class_names:
        n, recall = res["per_class_n"][name], res["recall"][name]
        print(f"  {name:<11} {n:>5} images, rappel {recall:.1%}")
    print("  matrice de confusion (lignes = vrai, colonnes = prédit) :")
    print("  " + " " * 11 + "".join(f"{n[:6]:>8}" for n in class_names))
    for name, row in zip(class_names, res["confusion_matrix"], strict=True):
        print(f"  {name:<11}" + "".join(f"{v:>8}" for v in row))
    t = res["tumor_predicted_notumor"]
    rate = "n/a" if t["rate"] is None else f"{t['rate']:.1%}"
    print(f"  tumeurs prédites « notumor » : {t['count']}/{t['total']} ({rate})")
    h = res["notumor_predicted_tumor"]
    print(f"  « notumor » prédites tumeur : {h['count']}/{h['total']}")
    for key, label in (("correct", "bonnes"), ("wrong", "mauvaises")):
        value = res[f"mean_confidence_{key}"]
        text = "n/a" if value is None else f"{value:.1%}"
        print(f"  confiance moyenne ({label} réponses) : {text}")


def main() -> None:
    """Charge le modèle, mesure les trois sous-ensembles, écrit le JSON, affiche le résumé."""
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    torch.manual_seed(0)
    config = load_config()
    path = config.training.checkpoint_path
    if not path.is_file():
        raise FileNotFoundError(f"Modèle introuvable : {path}")
    root = config.data.raw_dir.parent / "external" / "pmram"
    manifest = root / "manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError(f"Manifeste introuvable : {manifest} (lancez download_external.py)")
    with open(manifest, encoding="utf-8") as file:
        entries = json.load(file)["images"]

    device = get_device()
    checkpoint = load_checkpoint(path, device)
    class_names = list(checkpoint["class_names"])
    model = build_model(len(class_names), ModelConfig(pretrained=False, freeze_backbone=False))
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    transform = build_transforms(config.data, train=False)  # identique au test Kaggle

    report: dict = {"class_names": class_names, "subsets": {}}
    for name, subset in select_subsets(entries).items():
        report["subsets"][name] = evaluate_subset(
            model, root, subset, class_names, transform, device, config.data.batch_size
        )
    for name in SUBSET_NAMES:
        print_subset(TITLES[name], report["subsets"][name], class_names)

    print(
        "\nAttention : toutes les images PMRAM sont en 512x512 RGB, y compris « notumor ». "
        "À l'entraînement, « notumor » n'était presque jamais en 512x512 : le nombre de "
        "« notumor » prédites tumeur teste le raccourci de taille dans le sens inverse."
    )
    out = config.training.figures_dir / "external_pmram.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)
    print(f"\nRapport enregistré dans {out}")


if __name__ == "__main__":
    main()
