"""Chargement du modèle au démarrage de l'API (une seule fois)."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from brain_mri.predict import CONFIG_NAME, WEIGHTS_NAME, Predictor

from app.config import Settings

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class LoadedModel:
    """Le modèle prêt à prédire, avec sa description (scores, prétraitement)."""

    predictor: Predictor  # modèle en eval(), prêt à prédire
    description: dict  # contenu de config.json (classes, scores de test, etc.)
    version: str  # origine du modèle, affichée par /health


def load_model(settings: Settings) -> LoadedModel:
    """Charge le modèle depuis un dossier local ou depuis Hugging Face.

    Args:
        settings: Réglages de l'API.

    Returns:
        Le modèle chargé.

    Raises:
        FileNotFoundError: Si le dossier local ne contient pas les fichiers attendus.
    """
    if settings.model_dir is not None:
        folder, version = settings.model_dir, f"local:{settings.model_dir.name}"
    else:
        from huggingface_hub import hf_hub_download  # import ici : inutile en mode local

        # Les deux fichiers sont téléchargés dans le même dossier de cache.
        paths = [
            Path(hf_hub_download(settings.model_repo_id, name))
            for name in (WEIGHTS_NAME, CONFIG_NAME)
        ]
        folder, version = paths[0].parent, f"hub:{settings.model_repo_id}"

    predictor = Predictor.from_folder(folder)
    description = json.loads((folder / CONFIG_NAME).read_text(encoding="utf-8"))
    LOGGER.info("Modèle chargé (%s)", version)
    return LoadedModel(predictor, description, version)
