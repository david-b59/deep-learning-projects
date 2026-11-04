"""Réglages de l'API, lus dans les variables d'environnement (jamais écrits en dur).

Variables (toutes facultatives) :
    - ``MODEL_DIR`` : dossier exporté (``model.safetensors`` + ``config.json``). S'il est
      défini, il est utilisé en priorité (développement local, Docker avec volume).
    - ``MODEL_REPO_ID`` : sinon, dépôt Hugging Face d'où télécharger le modèle.
    - ``CORS_ORIGIN`` : adresse du site autorisée à appeler l'API (le front).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

DEFAULT_REPO_ID = "david-b59/brain-tumor-classifier"
DEFAULT_CORS_ORIGIN = "http://localhost:5173"  # adresse du front en développement
MAX_UPLOAD_BYTES = 5 * 1024 * 1024  # taille maximale d'une image envoyée : 5 Mo
MULTIPART_MARGIN_BYTES = 100 * 1024  # marge pour l'enveloppe multipart autour du fichier
MAX_IMAGE_PIXELS = 25_000_000  # au-delà, l'image est refusée (protège la mémoire)

# Le fichier .env se trouve à la racine du projet brain-tumor-classifier/.
PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    """Réglages de l'API."""

    model_dir: Path | None  # dossier local du modèle (prioritaire s'il est défini)
    model_repo_id: str  # dépôt Hugging Face (utilisé sans model_dir)
    cors_origin: str  # origine autorisée par CORS


def load_settings() -> Settings:
    """Lit les réglages dans l'environnement (et dans ``.env`` s'il existe).

    Returns:
        Les réglages de l'API.
    """
    load_dotenv(PROJECT_ROOT / ".env")  # n'écrase pas les variables déjà définies
    model_dir = os.environ.get("MODEL_DIR") or None  # chaîne vide = non défini
    return Settings(
        model_dir=Path(model_dir) if model_dir else None,
        model_repo_id=os.environ.get("MODEL_REPO_ID") or DEFAULT_REPO_ID,
        cors_origin=os.environ.get("CORS_ORIGIN") or DEFAULT_CORS_ORIGIN,
    )
