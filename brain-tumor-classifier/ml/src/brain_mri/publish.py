"""Export et publication du modèle sur Hugging Face Hub.

Deux étapes, volontairement séparées :
    1. **Exporter** ``models/best.pt`` vers un dossier propre (``models/export/``) :
       ``model.safetensors`` (les poids), ``model.onnx`` (le même modèle pour le navigateur,
       avec la carte Grad-CAM : voir :mod:`brain_mri.export_onnx`), ``config.json`` (classes,
       prétraitement, scores) et ``README.md`` (la model card, copiée depuis
       ``docs/model_card.md``).
    2. **Publier** ce dossier sur le Hub, seulement si on le demande avec ``--push``.

Pourquoi ``safetensors`` et pas ``.pt`` : un fichier ``.pt`` est un « pickle », qui peut
exécuter du code à l'ouverture. ``safetensors`` ne contient que des nombres : on peut le
partager et l'ouvrir sans risque (voir ton cours 7).

Le jeton Hugging Face vient de la variable d'environnement ``HF_TOKEN`` (fichier ``.env``,
jamais commité).

Utilisation (depuis ``brain-tumor-classifier/ml/``) ::

    uv run python -m brain_mri.publish            # exporte seulement (safetensors + ONNX)
    uv run python -m brain_mri.publish --push     # exporte puis publie
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import truststore
from dotenv import load_dotenv
from safetensors.torch import save_file

from brain_mri.config import PROJECT_ROOT, Config, load_config
from brain_mri.predict import CONFIG_NAME, DISCLAIMER, WEIGHTS_NAME
from brain_mri.train import load_checkpoint

if TYPE_CHECKING:
    from huggingface_hub import HfApi

LOGGER = logging.getLogger(__name__)

DEFAULT_EXPORT_DIR = PROJECT_ROOT / "models" / "export"  # dossier exporté (hors git)
DEFAULT_MODEL_CARD = PROJECT_ROOT / "docs" / "model_card.md"  # fiche du modèle (versionnée)
DEFAULT_REPO_NAME = "brain-tumor-classifier"  # nom du dépôt sur le Hub


def export_model(
    config: Config,
    out_dir: Path = DEFAULT_EXPORT_DIR,
    model_card: Path = DEFAULT_MODEL_CARD,
    with_onnx: bool = False,
) -> Path:
    """Exporte le meilleur modèle vers un dossier prêt à publier.

    Args:
        config: Configuration complète.
        out_dir: Dossier de sortie (créé si besoin, écrasé s'il existe déjà).
        model_card: Fichier Markdown de la fiche du modèle, copié en ``README.md``.
        with_onnx: Si True, écrit aussi ``model.onnx`` (la démo du site le télécharge).
            Faux par défaut : l'export ONNX prend une vingtaine de secondes.

    Returns:
        Le dossier exporté.

    Raises:
        FileNotFoundError: Si le modèle entraîné ou la fiche du modèle est absent.
    """
    checkpoint_path = config.training.checkpoint_path
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Modèle introuvable : {checkpoint_path} (lance d'abord train)")
    if not model_card.is_file():
        raise FileNotFoundError(f"Fiche du modèle introuvable : {model_card}")

    checkpoint = load_checkpoint(checkpoint_path)
    shutil.rmtree(out_dir, ignore_errors=True)  # repart d'un dossier vide : pas de fichier orphelin
    out_dir.mkdir(parents=True)

    # safetensors veut des tenseurs contigus en mémoire, sans partage.
    tensors = {name: t.contiguous() for name, t in checkpoint["state_dict"].items()}
    save_file(tensors, out_dir / WEIGHTS_NAME)

    description = {
        "architecture": "resnet50",
        "class_names": checkpoint["class_names"],
        "preprocessing": {key: checkpoint[key] for key in ("image_size", "mean", "std")},
        "selected_on_validation": {
            key: checkpoint[key]
            for key in ("phase", "epoch", "val_loss", "val_acc")
            if key in checkpoint
        },
        "disclaimer": DISCLAIMER,
    }
    results = config.training.figures_dir / "evaluation.json"
    if results.is_file():  # scores du test, s'ils ont été calculés (étape 4)
        evaluation = json.loads(results.read_text(encoding="utf-8"))
        description["test"] = {
            "accuracy": evaluation["accuracy"],
            "per_class": evaluation["per_class"],
            "confusion_matrix": evaluation["confusion_matrix"],
        }
    (out_dir / CONFIG_NAME).write_text(
        json.dumps(description, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if with_onnx:
        # Import ici : onnx/onnxruntime viennent de l'extra « train », inutiles sans `with_onnx`.
        from brain_mri.export_onnx import FP32_NAME, export_onnx, load_predictor

        predictor = load_predictor(config)
        export_onnx(predictor.model, out_dir / FP32_NAME, predictor.image_size)
    shutil.copyfile(
        model_card, out_dir / "README.md"
    )  # le Hub affiche README.md comme page du modèle
    LOGGER.info("Modèle exporté dans %s", out_dir)
    return out_dir


def resolve_repo_id(api: HfApi) -> str:
    """Détermine l'identifiant du dépôt : ``HF_REPO_ID`` s'il est défini, sinon ``compte/nom``.

    Args:
        api: Client ``HfApi`` déjà authentifié.

    Returns:
        L'identifiant, par exemple ``david-b59/brain-tumor-classifier``.
    """
    explicit = os.environ.get("HF_REPO_ID")
    if explicit:
        return explicit
    return f"{api.whoami()['name']}/{DEFAULT_REPO_NAME}"


def push_to_hub(folder: Path, repo_id: str | None = None, private: bool = False) -> str:
    """Publie un dossier exporté sur Hugging Face Hub.

    Args:
        folder: Dossier créé par :func:`export_model`.
        repo_id: Dépôt cible (par défaut : voir :func:`resolve_repo_id`).
        private: Si True, le dépôt est créé privé (sans effet s'il existe déjà).

    Returns:
        L'adresse web du dépôt.

    Raises:
        RuntimeError: Si ``HF_TOKEN`` n'est pas défini.
    """
    from huggingface_hub import HfApi

    load_dotenv(PROJECT_ROOT / ".env")  # lit le jeton dans .env (jamais dans le code)
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN absent : renseigne-le dans le fichier .env")

    api = HfApi(token=token)
    repo_id = repo_id or resolve_repo_id(api)
    api.create_repo(repo_id, repo_type="model", private=private, exist_ok=True)
    api.upload_folder(
        folder_path=str(folder),
        repo_id=repo_id,
        repo_type="model",
        commit_message="Publication du modèle (ResNet-50, 4 classes d'IRM)",
    )
    url = f"https://huggingface.co/{repo_id}"
    LOGGER.info("Modèle publié : %s", url)
    return url


def main() -> None:
    """Point d'entrée en ligne de commande : exporte, puis publie si ``--push`` est demandé."""
    parser = argparse.ArgumentParser(description="Exporte (et publie) le modèle.")
    parser.add_argument(
        "--push", action="store_true", help="Publie sur Hugging Face après l'export"
    )
    parser.add_argument("--private", action="store_true", help="Dépôt privé (avec --push)")
    parser.add_argument("--repo-id", default=None, help="Dépôt cible, ex. compte/nom")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    truststore.inject_into_ssl()
    folder = export_model(load_config(), with_onnx=True)
    print(f"Dossier exporté : {folder}")
    for file in sorted(folder.iterdir()):
        print(f"  {file.name:<18} {file.stat().st_size / 1e6:8.2f} Mo")
    if args.push:
        print(f"Publié : {push_to_hub(folder, args.repo_id, args.private)}")
    else:
        print("Rien n'est publié (ajoute --push pour l'envoyer sur Hugging Face).")


if __name__ == "__main__":
    main()
