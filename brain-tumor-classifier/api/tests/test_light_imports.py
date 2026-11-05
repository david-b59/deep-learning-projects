"""Vérifie que l'API démarre sans importer les bibliothèques lourdes d'entraînement."""

from __future__ import annotations

import subprocess
import sys

# Réservés à l'entraînement (ml/). matplotlib n'est pas dans la liste : Captum 0.9 l'importe
# lui-même au chargement et l'impose comme dépendance, on ne peut donc pas l'éviter tant que
# l'API utilise Captum pour le Grad-CAM.
INTERDITS = ("mlflow", "sklearn", "pandas")


def test_app_main_n_importe_pas_les_bibliotheques_d_entrainement() -> None:
    """``import app.main`` ne doit charger ni mlflow, ni sklearn, ni pandas."""
    code = (
        "import app.main, sys; "
        f"found = [m for m in {INTERDITS!r} if m in sys.modules]; "
        "print(','.join(found)); sys.exit(1 if found else 0)"
    )
    # Sous-processus : un interpréteur neuf, non pollué par les autres tests
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, (
        f"Modules d'entraînement importés : {result.stdout} {result.stderr[-500:]}"
    )
