"""Check the dataset for duplicate images and write a report.

Compares the Training and Testing folders by file content (SHA-256), prints a
numeric summary and saves ``data/processed/duplicates_report.json``. The data
pipeline reads this report to remove the leaked images from the test set.

Usage (from ``brain-tumor-classifier/ml/``)::

    uv run python scripts/check_duplicates.py
"""

from __future__ import annotations

import logging

from brain_mri.config import load_config
from brain_mri.duplicates import find_duplicates, save_report


def main() -> None:
    """Run the duplicate check with the paths of ``configs/train.yaml``."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config = load_config().data
    report = find_duplicates(config.raw_dir)
    path = save_report(report, config.processed_dir)

    print(f"Training images                          : {report.n_train}")
    print(f"Testing images                           : {report.n_test}")
    print(f"Test images also present in Training     : {len(report.test_files_in_train)}")
    print(f"Groups of identical images in Training   : {len(report.train_groups)}")
    print(f"Groups of identical images in Testing    : {len(report.test_groups)}")
    print(f"Report saved to {path}")


if __name__ == "__main__":
    main()
