"""Download the Brain Tumor MRI Dataset from Kaggle into ``data/raw/``.

Credentials are read from the environment (``KAGGLE_USERNAME`` and
``KAGGLE_API_TOKEN``), usually loaded from ``brain-tumor-classifier/.env``.
Nothing is written to git: ``data/`` is ignored.

Dataset: Masoud Nickparvar, "Brain Tumor MRI Dataset", licensed CC BY 4.0.
https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

Usage (from ``brain-tumor-classifier/ml/``)::

    uv run python scripts/download_data.py
"""

from __future__ import annotations

from pathlib import Path

import truststore
from dotenv import load_dotenv

# Use the operating system certificate store, so downloads also work behind
# networks that re-sign HTTPS traffic. Certificate checks stay enabled.
truststore.inject_into_ssl()

DATASET = "masoudnickparvar/brain-tumor-mri-dataset"
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # brain-tumor-classifier/
RAW_DIR = PROJECT_ROOT / "data" / "raw"
EXPECTED_SPLITS = ("Training", "Testing")


def is_already_downloaded(raw_dir: Path) -> bool:
    """Return True if both ``Training`` and ``Testing`` folders already exist."""
    return all((raw_dir / split).is_dir() for split in EXPECTED_SPLITS)


def download(raw_dir: Path = RAW_DIR) -> Path:
    """Download and unzip the dataset into ``raw_dir`` (skipped if already there).

    Args:
        raw_dir: Destination folder.

    Returns:
        The destination folder.
    """
    if is_already_downloaded(raw_dir):
        print(f"Already downloaded in {raw_dir}, nothing to do.")
        return raw_dir

    # Load the secrets from .env BEFORE importing kaggle: the client reads the
    # environment when it authenticates.
    load_dotenv(PROJECT_ROOT / ".env")
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()  # raises a clear error if the token is missing or invalid

    raw_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {DATASET} into {raw_dir} ...")
    api.dataset_download_files(DATASET, path=str(raw_dir), unzip=True, quiet=False)

    if not is_already_downloaded(raw_dir):
        raise RuntimeError(f"Expected folders {EXPECTED_SPLITS} not found in {raw_dir}")
    print("Done.")
    return raw_dir


if __name__ == "__main__":
    download()
