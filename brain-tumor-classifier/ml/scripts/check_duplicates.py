"""Vérifie les images en double dans le jeu de données et écrit un rapport.

Compare les dossiers Training et Testing par contenu de fichier (SHA-256), affiche
un résumé chiffré et enregistre ``data/processed/duplicates_report.json``. Le
pipeline de données lit ce rapport pour écarter les images en double.

Utilisation (depuis ``brain-tumor-classifier/ml/``) ::

    uv run python scripts/check_duplicates.py
"""

from __future__ import annotations

import logging

from brain_mri.config import load_config
from brain_mri.duplicates import find_duplicates, save_report


def main() -> None:
    """Lance la vérification des doublons avec les chemins de ``configs/train.yaml``."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config = load_config().data
    report = find_duplicates(config.raw_dir)
    path = save_report(report, config.processed_dir)

    print(f"Images dans Training                              : {report.n_train}")
    print(f"Images dans Testing                               : {report.n_test}")
    print(f"Images de Testing aussi présentes dans Training   : {len(report.test_files_in_train)}")
    print(f"Groupes d'images identiques dans Training         : {len(report.train_groups)}")
    print(f"Groupes d'images identiques dans Testing          : {len(report.test_groups)}")
    print(f"Rapport enregistré dans {path}")


if __name__ == "__main__":
    main()
