"""Télécharge le Brain Tumor MRI Dataset depuis Kaggle dans ``data/raw/``.

Les identifiants sont lus dans l'environnement (``KAGGLE_USERNAME`` et
``KAGGLE_API_TOKEN``), en général chargés depuis ``brain-tumor-classifier/.env``.
Rien n'est écrit dans git : le dossier ``data/`` est ignoré.

Jeu de données : Masoud Nickparvar, « Brain Tumor MRI Dataset », licence CC BY 4.0.
https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

Utilisation (depuis ``brain-tumor-classifier/ml/``) ::

    uv run python scripts/download_data.py
"""

from __future__ import annotations

from pathlib import Path

import truststore
from dotenv import load_dotenv

# Utilise le magasin de certificats du système d'exploitation, pour que le
# téléchargement fonctionne aussi derrière un réseau qui re-signe le trafic HTTPS.
# La vérification des certificats reste activée.
truststore.inject_into_ssl()

DATASET = "masoudnickparvar/brain-tumor-mri-dataset"  # identifiant Kaggle du jeu de données
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # brain-tumor-classifier/
RAW_DIR = PROJECT_ROOT / "data" / "raw"
EXPECTED_SPLITS = ("Training", "Testing")  # dossiers attendus après décompression


def is_already_downloaded(raw_dir: Path) -> bool:
    """Indique si les dossiers ``Training`` et ``Testing`` existent déjà."""
    return all((raw_dir / split).is_dir() for split in EXPECTED_SPLITS)


def download(raw_dir: Path = RAW_DIR) -> Path:
    """Télécharge et décompresse le jeu de données dans ``raw_dir`` (sauf s'il y est déjà).

    Args:
        raw_dir: Dossier de destination.

    Returns:
        Le dossier de destination.

    Raises:
        RuntimeError: Si les dossiers attendus sont absents après le téléchargement.
    """
    if is_already_downloaded(raw_dir):
        print(f"Déjà téléchargé dans {raw_dir}, rien à faire.")
        return raw_dir

    # On charge les secrets du .env AVANT d'importer kaggle : le client lit
    # l'environnement au moment de s'authentifier.
    load_dotenv(PROJECT_ROOT / ".env")
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()  # lève une erreur claire si le jeton est absent ou invalide

    raw_dir.mkdir(parents=True, exist_ok=True)
    print(f"Téléchargement de {DATASET} dans {raw_dir} ...")
    api.dataset_download_files(DATASET, path=str(raw_dir), unzip=True, quiet=False)

    if not is_already_downloaded(raw_dir):
        raise RuntimeError(f"Dossiers attendus {EXPECTED_SPLITS} introuvables dans {raw_dir}")
    print("Terminé.")
    return raw_dir


if __name__ == "__main__":
    download()
