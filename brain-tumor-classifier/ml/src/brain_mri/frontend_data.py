"""Données du site : copie les scores du jeu de test dans un JSON lu par la page « Résultats ».

Le site public n'a plus d'API : les scores sont intégrés à la compilation. Ce script lit
``docs/figures/evaluation.json`` (écrit par :mod:`brain_mri.evaluate`) et écrit
``frontend/src/lib/metrics.json`` avec exactement les champs attendus par le type
``MetricsSummary`` du site : ``accuracy``, ``class_names``, ``per_class``
(``precision``, ``recall``, ``f1``, ``support``) et ``confusion_matrix``.

Utilisation (depuis ``brain-tumor-classifier/ml/``) ::

    uv run python -m brain_mri.frontend_data
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

LOGGER = logging.getLogger(__name__)

# Dossier du projet (brain-tumor-classifier/) : ce fichier est dans ml/src/brain_mri/
PROJECT_DIR = Path(__file__).resolve().parents[3]
DEFAULT_EVALUATION = PROJECT_DIR / "docs" / "figures" / "evaluation.json"
DEFAULT_OUTPUT = PROJECT_DIR / "frontend" / "src" / "lib" / "metrics.json"

SCORE_FIELDS = ("precision", "recall", "f1", "support")  # champs gardés pour chaque classe


def build_metrics_summary(evaluation: dict) -> dict:
    """Extrait du fichier d'évaluation les seuls champs utiles à la page « Résultats ».

    Args:
        evaluation: Contenu de ``evaluation.json``.

    Returns:
        Un dictionnaire ``accuracy``, ``class_names``, ``per_class``, ``confusion_matrix``.

    Raises:
        ValueError: Si un champ manque ou si la matrice de confusion n'a pas la taille attendue.
    """
    for key in ("accuracy", "per_class", "confusion_matrix"):
        if key not in evaluation:
            raise ValueError(f"Champ « {key} » absent du fichier d'évaluation.")
    class_names = list(evaluation["per_class"])  # l'ordre des classes = celui des indices
    matrix = evaluation["confusion_matrix"]
    if len(matrix) != len(class_names) or any(len(row) != len(class_names) for row in matrix):
        raise ValueError("La matrice de confusion ne correspond pas au nombre de classes.")
    per_class = {
        name: {field: evaluation["per_class"][name][field] for field in SCORE_FIELDS}
        for name in class_names
    }
    return {
        "accuracy": evaluation["accuracy"],
        "class_names": class_names,
        "per_class": per_class,
        "confusion_matrix": matrix,
    }


def write_metrics_json(evaluation_path: Path, output_path: Path) -> dict:
    """Lit ``evaluation.json`` et écrit le JSON du site.

    Args:
        evaluation_path: Fichier d'évaluation à lire.
        output_path: Fichier JSON à écrire (le dossier est créé au besoin).

    Returns:
        Le résumé écrit.

    Raises:
        FileNotFoundError: Si le fichier d'évaluation n'existe pas.
    """
    if not evaluation_path.is_file():
        raise FileNotFoundError(f"Fichier introuvable : {evaluation_path}")
    summary = build_metrics_summary(json.loads(evaluation_path.read_text(encoding="utf-8")))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    """Point d'entrée en ligne de commande."""
    parser = argparse.ArgumentParser(description="Écrit les scores du site (metrics.json).")
    parser.add_argument("--evaluation", type=Path, default=DEFAULT_EVALUATION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    summary = write_metrics_json(args.evaluation, args.output)
    LOGGER.info("Scores écrits dans %s (accuracy %.4f)", args.output, summary["accuracy"])


if __name__ == "__main__":
    main()
