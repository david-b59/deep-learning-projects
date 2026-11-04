"""Prédiction sur une IRM : classe, probabilités et Grad-CAM (phase E, utilisation).

C'est ce module que l'API (étape 7) appellera. Il charge le modèle **une seule fois** puis
prédit autant d'images qu'on veut.

Trois façons de charger le modèle :
    - :meth:`Predictor.from_checkpoint` : le fichier ``best.pt`` de l'entraînement ;
    - :meth:`Predictor.from_folder` : un dossier exporté par :mod:`brain_mri.publish`
      (``model.safetensors`` + ``config.json``) ;
    - :meth:`Predictor.from_hub` : le même dossier, téléchargé depuis Hugging Face.

Utilisation (depuis ``brain-tumor-classifier/ml/``) ::

    uv run python -m brain_mri.predict chemin/vers/image.jpg
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
import truststore
from PIL import Image
from safetensors.torch import load_file
from torch import nn

from brain_mri.config import DataConfig, ModelConfig, load_config
from brain_mri.data import build_transforms
from brain_mri.gradcam import gradcam_captum, upsample
from brain_mri.model import build_model
from brain_mri.train import get_device, load_checkpoint

LOGGER = logging.getLogger(__name__)

DISCLAIMER = "Démonstration technique, pas un dispositif médical."
WEIGHTS_NAME = "model.safetensors"  # poids du modèle (format sans risque, pas de pickle)
CONFIG_NAME = "config.json"  # classes, prétraitement, scores
OVERLAY_ALPHA = 0.45  # opacité de la carte de chaleur sur l'IRM


@dataclass(frozen=True)
class Prediction:
    """Résultat de la prédiction d'une image."""

    predicted_class: str  # classe la plus probable
    probabilities: dict[str, float]  # probabilité de chaque classe (somme = 1)
    gradcam_png_base64: str  # IRM + carte Grad-CAM superposée, PNG encodé en base64
    disclaimer: str = DISCLAIMER  # rappel : outil de démonstration


def overlay_heatmap(
    image: np.ndarray, heatmap: np.ndarray, alpha: float = OVERLAY_ALPHA
) -> Image.Image:
    """Superpose une carte de chaleur (palette « jet ») à une image.

    Args:
        image: IRM ``(H, W, 3)`` en valeurs 0-1.
        heatmap: Carte ``(H, W)`` en valeurs 0-1.
        alpha: Opacité de la carte (0 = invisible, 1 = remplace l'image).

    Returns:
        L'image superposée, en RGB.
    """
    colors = matplotlib.colormaps["jet"](heatmap)[..., :3]  # carte -> couleurs RGB (sans l'alpha)
    blended = (1 - alpha) * image + alpha * colors
    return Image.fromarray((blended.clip(0, 1) * 255).astype(np.uint8))


def encode_png_base64(image: Image.Image) -> str:
    """Encode une image PIL en PNG puis en texte base64 (pour la renvoyer dans du JSON).

    Args:
        image: L'image à encoder.

    Returns:
        Le PNG encodé en base64 (texte).
    """
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


class Predictor:
    """Modèle chargé, prêt à prédire des images (à créer une seule fois).

    Args:
        model: Le ResNet-50 avec ses poids entraînés.
        class_names: Noms des classes, dans l'ordre des indices.
        data_config: Prétraitement attendu (taille, moyenne, écart-type).
        device: CPU ou GPU.
    """

    def __init__(
        self,
        model: nn.Module,
        class_names: list[str],
        data_config: DataConfig,
        device: torch.device,
    ) -> None:
        """Range le modèle en mode évaluation et prépare le prétraitement."""
        self.model = model.to(device).eval()  # eval : indispensable (BatchNorm, Grad-CAM)
        self.class_names = class_names
        self.image_size = data_config.image_size
        self.transform = build_transforms(data_config, train=False)  # sans augmentation
        self.device = device

    @staticmethod
    def _build(
        state_dict: dict, class_names: list[str], pre: dict, device: torch.device
    ) -> Predictor:
        """Construit un Predictor à partir de poids et du prétraitement enregistrés.

        Args:
            state_dict: Poids du modèle.
            class_names: Noms des classes.
            pre: Prétraitement : ``image_size``, ``mean``, ``std``.
            device: CPU ou GPU.

        Returns:
            Le Predictor prêt à l'emploi.
        """
        model = build_model(len(class_names), ModelConfig(pretrained=False, freeze_backbone=False))
        model.load_state_dict(state_dict)
        data_config = DataConfig(
            image_size=pre["image_size"], mean=tuple(pre["mean"]), std=tuple(pre["std"])
        )
        return Predictor(model, class_names, data_config, device)

    @classmethod
    def from_checkpoint(cls, path: Path, device: torch.device | None = None) -> Predictor:
        """Charge le fichier ``best.pt`` produit par l'entraînement.

        Args:
            path: Chemin du fichier.
            device: CPU ou GPU (par défaut : détecté).

        Returns:
            Le Predictor.
        """
        device = device or get_device()
        checkpoint = load_checkpoint(path, device)
        pre = {key: checkpoint[key] for key in ("image_size", "mean", "std")}
        return cls._build(checkpoint["state_dict"], checkpoint["class_names"], pre, device)

    @classmethod
    def from_folder(cls, folder: Path, device: torch.device | None = None) -> Predictor:
        """Charge un dossier exporté (``model.safetensors`` + ``config.json``).

        Args:
            folder: Dossier contenant les deux fichiers.
            device: CPU ou GPU (par défaut : détecté).

        Returns:
            Le Predictor.

        Raises:
            FileNotFoundError: Si un des deux fichiers manque.
        """
        device = device or get_device()
        weights, config_file = folder / WEIGHTS_NAME, folder / CONFIG_NAME
        for file in (weights, config_file):
            if not file.is_file():
                raise FileNotFoundError(f"Fichier introuvable : {file}")
        config = json.loads(config_file.read_text(encoding="utf-8"))
        return cls._build(
            load_file(weights), config["class_names"], config["preprocessing"], device
        )

    @classmethod
    def from_hub(
        cls, repo_id: str, device: torch.device | None = None, token: str | None = None
    ) -> Predictor:
        """Télécharge le modèle depuis Hugging Face (mis en cache) puis le charge.

        Args:
            repo_id: Identifiant du dépôt, par exemple ``david-b59/brain-tumor-classifier``.
            device: CPU ou GPU (par défaut : détecté).
            token: Jeton Hugging Face (inutile pour un dépôt public).

        Returns:
            Le Predictor.
        """
        from huggingface_hub import hf_hub_download

        paths = [
            Path(hf_hub_download(repo_id, name, token=token))
            for name in (WEIGHTS_NAME, CONFIG_NAME)
        ]
        return cls.from_folder(paths[0].parent, device)

    def predict(self, image: Image.Image) -> Prediction:
        """Prédit la classe d'une IRM et explique la décision avec Grad-CAM.

        Args:
            image: L'image (n'importe quel mode : elle est convertie en RGB).

        Returns:
            La classe, les probabilités, et l'image Grad-CAM superposée en PNG base64.
        """
        image = image.convert("RGB")  # les IRM sont parfois en niveaux de gris
        x = self.transform(image).unsqueeze(0).to(self.device)  # ajoute la dimension du lot
        with torch.no_grad():
            probs = F.softmax(self.model(x), dim=1)[0].cpu()  # scores -> probabilités
        index = int(probs.argmax())

        # Grad-CAM a besoin des gradients : pas de no_grad ici. On explique la classe prédite.
        heatmap = upsample(gradcam_captum(self.model, x, index).cpu(), self.image_size)
        display = (
            np.asarray(image.resize((self.image_size, self.image_size)), dtype=np.float32) / 255
        )
        overlay = overlay_heatmap(display, heatmap)
        return Prediction(
            predicted_class=self.class_names[index],
            probabilities={name: float(p) for name, p in zip(self.class_names, probs, strict=True)},
            gradcam_png_base64=encode_png_base64(overlay),
        )


def main() -> None:
    """Point d'entrée en ligne de commande : prédit une image et affiche le résultat."""
    parser = argparse.ArgumentParser(description="Prédit la classe d'une IRM.")
    parser.add_argument("image", type=Path, help="Chemin de l'image (JPG ou PNG)")
    parser.add_argument("--folder", type=Path, default=None, help="Dossier exporté (sinon best.pt)")
    parser.add_argument(
        "--save-overlay", type=Path, default=None, help="Enregistre le Grad-CAM (PNG)"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    truststore.inject_into_ssl()
    config = load_config()
    if args.folder:
        predictor = Predictor.from_folder(args.folder)
    else:
        predictor = Predictor.from_checkpoint(config.training.checkpoint_path)

    result = predictor.predict(Image.open(args.image))
    print(f"\nClasse prédite : {result.predicted_class}")
    for name, p in sorted(result.probabilities.items(), key=lambda item: -item[1]):
        print(f"  {name:<11} {p:6.1%}")
    print(f"\n{result.disclaimer}")
    if args.save_overlay:
        args.save_overlay.write_bytes(base64.b64decode(result.gradcam_png_base64))
        print(f"Grad-CAM enregistré : {args.save_overlay}")


if __name__ == "__main__":
    main()
