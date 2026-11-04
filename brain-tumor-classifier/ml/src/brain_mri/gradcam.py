"""Interprétabilité : Grad-CAM (phase E du pipeline).

Grad-CAM répond à la question « quelles zones de l'image ont poussé le modèle vers cette
classe ? ». Il utilise les 2048 cartes de ``layer4`` (7 x 7) : chaque carte est pondérée
par la moyenne du gradient du score de la classe, puis on additionne, on garde les valeurs
positives (ReLU) et on normalise entre 0 et 1.

Ce module contient les deux versions de ton cours 6 :
    - :func:`gradcam_manual` : à la main, avec un hook sur ``layer4`` ;
    - :func:`gradcam_captum` : avec ``LayerGradCam`` de Captum (la version utilisée en
      pratique). Les deux cartes sont identiques après normalisation.

Puis la galerie : quelques bonnes prédictions et quelques erreurs du jeu de test, la carte
superposée à l'IRM.

Écarts avec le cours (corrections déjà identifiées) :
    - le modèle est toujours en ``eval()`` (sinon BatchNorm fausse la carte) ;
    - la carte 7 x 7 est agrandie à 224 x 224 et **superposée** à l'image ;
    - la ReLU est appliquée par Captum (``relu_attributions=True``).

Utilisation (depuis ``brain-tumor-classifier/ml/``, après ``train`` et ``evaluate``) ::

    uv run python -m brain_mri.gradcam
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # pas de fenêtre : les figures sont écrites dans des fichiers
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import truststore
from captum.attr import LayerAttribution, LayerGradCam
from PIL import Image
from torch import nn

from brain_mri.config import Config, ModelConfig, load_config
from brain_mri.data import build_dataloaders, build_transforms
from brain_mri.model import build_model
from brain_mri.train import get_device, load_checkpoint

LOGGER = logging.getLogger(__name__)

GALLERY_NAME = "gradcam_gallery.png"  # nom de la galerie dans le dossier des figures


def load_model(config: Config, device: torch.device) -> tuple[nn.Module, list[str]]:
    """Charge le meilleur modèle en mode évaluation.

    Args:
        config: Configuration complète.
        device: CPU ou GPU.

    Returns:
        ``(modèle, noms des classes)``.

    Raises:
        FileNotFoundError: Si le fichier du modèle n'existe pas.
    """
    path = config.training.checkpoint_path
    if not path.is_file():
        raise FileNotFoundError(f"Modèle introuvable : {path} (lance d'abord train)")
    checkpoint = load_checkpoint(path, device)
    model = build_model(
        len(checkpoint["class_names"]), ModelConfig(pretrained=False, freeze_backbone=False)
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()  # indispensable : sinon BatchNorm utilise le lot et fausse la carte
    return model, checkpoint["class_names"]


def load_image(path: Path, config: Config) -> tuple[torch.Tensor, np.ndarray]:
    """Charge une IRM : tenseur normalisé pour le modèle et image 224 x 224 pour l'affichage.

    Args:
        path: Chemin de l'image.
        config: Configuration (taille, moyenne et écart-type du prétraitement).

    Returns:
        ``(tenseur (1, 3, H, W), image (H, W, 3) en valeurs 0-1)``.
    """
    image = Image.open(path).convert("RGB")  # les IRM sont parfois en niveaux de gris
    tensor = build_transforms(config.data, train=False)(image).unsqueeze(0)  # ajoute le lot
    size = (config.data.image_size, config.data.image_size)
    display = np.asarray(image.resize(size), dtype=np.float32) / 255.0  # image sans normalisation
    return tensor, display


def _normalize(heatmap: torch.Tensor) -> torch.Tensor:
    """Ramène une carte entre 0 et 1 en la divisant par son maximum (carte nulle : inchangée).

    Args:
        heatmap: Carte à valeurs positives.

    Returns:
        La carte normalisée.
    """
    peak = heatmap.max()
    return heatmap / peak if peak > 0 else heatmap


def gradcam_manual(
    model: nn.Module, x: torch.Tensor, target: int, layer: nn.Module | None = None
) -> torch.Tensor:
    """Grad-CAM à la main, avec un hook (version du cours 6).

    Args:
        model: Le modèle (en ``eval()``).
        x: Une image prétraitée, forme ``(1, 3, H, W)``.
        target: Indice de la classe à expliquer.
        layer: Couche à observer (par défaut ``model.layer4``).

    Returns:
        La carte, forme ``(h, w)`` (7 x 7 pour layer4), valeurs entre 0 et 1.
    """
    layer = layer if layer is not None else model.layer4
    stored: dict[str, torch.Tensor] = {}

    def forward_hook(module: nn.Module, inputs: tuple, output: torch.Tensor) -> None:
        """Met de côté la sortie de la couche et demande à garder son gradient."""
        output.retain_grad()
        stored["activations"] = output

    handle = layer.register_forward_hook(forward_hook)
    try:
        # requires_grad : le gradient doit pouvoir remonter même si le corps est gelé.
        scores = model(x.clone().requires_grad_(True))
    finally:
        handle.remove()  # sinon le hook resterait actif pour tous les forwards suivants

    model.zero_grad()
    scores[0, target].backward()  # gradient du SCORE (logit) de la classe, pas de la probabilité
    activations = stored["activations"][0].detach()  # (2048, 7, 7)
    gradients = stored["activations"].grad[0]  # même forme
    weights = gradients.mean(dim=(1, 2))  # un poids par carte : moyenne de son gradient
    heatmap = (weights[:, None, None] * activations).mean(dim=0)  # somme pondérée des cartes
    return _normalize(torch.relu(heatmap))  # on ne garde que ce qui pousse vers la classe


def gradcam_captum(
    model: nn.Module, x: torch.Tensor, target: int, layer: nn.Module | None = None
) -> torch.Tensor:
    """Grad-CAM avec Captum (``LayerGradCam``).

    Args:
        model: Le modèle (en ``eval()``).
        x: Une image prétraitée, forme ``(1, 3, H, W)``.
        target: Indice de la classe à expliquer.
        layer: Couche à observer (par défaut ``model.layer4``).

    Returns:
        La carte, forme ``(h, w)``, valeurs entre 0 et 1.
    """
    layer = layer if layer is not None else model.layer4
    attribution = LayerGradCam(model, layer).attribute(x, target=target, relu_attributions=True)
    return _normalize(attribution.detach()[0, 0])  # (1, 1, 7, 7) -> (7, 7)


def upsample(heatmap: torch.Tensor, size: int) -> np.ndarray:
    """Agrandit une carte 7 x 7 à la taille de l'image (interpolation bilinéaire).

    Args:
        heatmap: Carte ``(h, w)``.
        size: Côté de l'image, en pixels.

    Returns:
        La carte ``(size, size)`` en NumPy, bornée entre 0 et 1.
    """
    big = LayerAttribution.interpolate(
        heatmap[None, None], (size, size), interpolate_mode="bilinear"
    )
    return big[0, 0].clamp(0, 1).numpy()


def predict_one(model: nn.Module, x: torch.Tensor) -> tuple[int, float]:
    """Donne la classe prédite et sa probabilité pour une image.

    Args:
        model: Le modèle (en ``eval()``).
        x: Image prétraitée ``(1, 3, H, W)``.

    Returns:
        ``(indice de la classe prédite, probabilité)``.
    """
    with torch.no_grad():
        probs = F.softmax(model(x), dim=1)[0]  # scores -> probabilités qui somment à 1
    index = int(probs.argmax())
    return index, float(probs[index])


def select_examples(
    samples: list[tuple[str, int]],
    class_names: list[str],
    results: dict,
    config: Config,
    n_confusions: int = 3,
    per_confusion: int = 2,
) -> list[dict]:
    """Choisit les images de la galerie : une bonne prédiction par classe, puis des erreurs.

    Le choix est aléatoire mais reproductible (graine de la configuration). Les erreurs
    viennent des confusions les plus fréquentes (par défaut 3 confusions, 2 images chacune).

    Args:
        samples: ``(chemin absolu, étiquette)`` de chaque image de test.
        class_names: Noms des classes.
        results: Contenu de ``evaluation.json``.
        config: Configuration complète.
        n_confusions: Nombre de confusions fréquentes dont on montre des erreurs.
        per_confusion: Nombre d'erreurs montrées pour chaque confusion.

    Returns:
        Une liste de dictionnaires ``{"path", "true", "kind"}`` (``kind`` = ``ok`` ou ``erreur``).
    """
    rng = random.Random(config.data.seed)
    raw = config.data.raw_dir
    wrong = {e["file"] for e in results["errors"]}
    chosen: list[dict] = []

    for index, name in enumerate(class_names):  # une bonne prédiction par classe
        good = [
            p
            for p, label in samples
            if label == index and Path(p).relative_to(raw).as_posix() not in wrong
        ]
        if good:  # une classe sans aucune bonne prédiction est simplement sautée
            chosen.append({"path": Path(rng.choice(good)), "true": name, "kind": "ok"})

    for pair in results["top_confusions"][:n_confusions]:  # les erreurs les plus fréquentes
        errors = [
            e
            for e in results["errors"]
            if e["true"] == pair["true"] and e["predicted"] == pair["predicted"]
        ]
        for error in rng.sample(errors, k=min(per_confusion, len(errors))):
            chosen.append({"path": raw / error["file"], "true": error["true"], "kind": "erreur"})
    return chosen


def overlay_figure(items: list[dict], path: Path) -> None:
    """Enregistre la galerie : pour chaque IRM, l'image seule puis la carte superposée.

    Args:
        items: Une entrée par image : ``image`` (H, W, 3), ``heatmap`` (H, W), ``title``.
        path: Fichier PNG à écrire (le dossier est créé si besoin).
    """
    pairs_per_row = 2
    rows = -(-len(items) // pairs_per_row)  # division par excès
    fig, axes = plt.subplots(
        rows, pairs_per_row * 2, figsize=(4 * pairs_per_row * 2 * 0.8, 3.4 * rows)
    )
    axes = np.atleast_2d(axes)
    for ax in axes.ravel():
        ax.axis("off")  # les cases inutilisées restent vides
    for i, item in enumerate(items):
        row, col = divmod(i, pairs_per_row)
        left, right = axes[row, col * 2], axes[row, col * 2 + 1]
        left.imshow(item["image"])
        right.imshow(item["image"])
        right.imshow(item["heatmap"], cmap="jet", alpha=0.45, vmin=0, vmax=1)  # superposition
        left.set_title("IRM", fontsize=9)
        right.set_title(item["title"], fontsize=8, color="green" if item["kind"] == "ok" else "red")
    fig.suptitle("Grad-CAM : où regarde le modèle (jeu de test)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))  # laisse de la place au titre général
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def build_gallery(config: Config) -> Path:
    """Construit et enregistre la galerie Grad-CAM (bonnes prédictions et erreurs).

    Args:
        config: Configuration complète.

    Returns:
        Le chemin de la galerie.

    Raises:
        FileNotFoundError: Si ``evaluation.json`` ou le modèle est absent.
    """
    figures_dir = config.training.figures_dir
    results_path = figures_dir / "evaluation.json"
    if not results_path.is_file():
        raise FileNotFoundError(f"{results_path} introuvable (lance d'abord evaluate)")
    results = json.loads(results_path.read_text(encoding="utf-8"))

    device = get_device()
    model, class_names = load_model(config, device)
    samples = build_dataloaders(config.data).test_loader.dataset.samples

    items = []
    for example in select_examples(samples, class_names, results, config):
        x, display = load_image(example["path"], config)
        predicted, probability = predict_one(model, x.to(device))
        heatmap = gradcam_captum(model, x.to(device), predicted).cpu()  # on explique la prédiction
        title = f"vrai : {example['true']}\nprédit : {class_names[predicted]} ({probability:.0%})"
        items.append(
            {
                "image": display,
                "heatmap": upsample(heatmap, config.data.image_size),
                "title": title,
                "kind": example["kind"],
            }
        )
    path = figures_dir / GALLERY_NAME
    overlay_figure(items, path)
    LOGGER.info("Galerie enregistrée : %s", path)
    return path


def main() -> None:
    """Point d'entrée en ligne de commande : construit la galerie Grad-CAM."""
    parser = argparse.ArgumentParser(description="Construit la galerie Grad-CAM.")
    parser.add_argument("--config", type=Path, default=None, help="Fichier YAML de configuration")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    truststore.inject_into_ssl()
    config = load_config(args.config) if args.config else load_config()
    print(f"Galerie : {build_gallery(config)}")


if __name__ == "__main__":
    main()
