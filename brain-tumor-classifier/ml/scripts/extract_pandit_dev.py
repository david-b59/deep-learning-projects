"""Extrait les coupes de Pandit « développement » : 40 patients disjoints du test (étape 11c-0).

Ce jeu sert à AJUSTER le détecteur de séquence T2 (régression logistique sur les caractéristiques
A1, voir ``ml/splits/guards_rules.md``). Il ne doit jamais contenir un des 30 patients tirés à
l'étape 11b-7 pour le test externe (``ml/splits/external_pandit.csv``, ``external_rules.md``).

Tirage : les 30 patients du test sont reproduits exactement (``download_pandit.choose_patients``,
graine 42, parmi ceux qui ont T1, T2 et masque) et comparés au manifeste du test ; puis 40
patients sont tirés (graine 42) parmi les autres candidats. Si un patient est écarté (grille du
masque ou du T2 différente de celle de la T1, trop peu de coupes saines), il est remplacé par le
suivant d'un second tirage de réserve (même graine, hors patients déjà tirés).

Extraction : mêmes règles que le test (``external_rules.md``), en réutilisant
``extract_external_slices.process_patient`` sans le modifier : coupe tumeur + coupes saines, mêmes
niveaux de coupe en T1c et en T2. Les volumes sont supprimés après chaque patient.

Sorties : PNG dans ``data/external/pandit_dev/`` (ignoré par git) et manifeste versionné
``ml/splits/pandit_dev.csv`` (mêmes colonnes que ``external_pandit.csv``).

Utilisation (depuis ``brain-tumor-classifier/ml/``) ::

    PYTHONUTF8=1 uv run python scripts/extract_pandit_dev.py
"""

from __future__ import annotations

import argparse
import csv
import logging
import random
import shutil
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # pour importer les scripts voisins

import download_pandit as dp  # noqa: E402
import extract_external_slices as ex  # noqa: E402

LOGGER = logging.getLogger("extract_pandit_dev")
ROOT = dp.PROJECT_ROOT
DEV_DIR = ROOT / "data" / "external" / "pandit_dev"
VOLUMES_DIR = DEV_DIR / "volumes"
MANIFEST = ROOT / "ml" / "splits" / "pandit_dev.csv"
TEST_MANIFEST = ROOT / "ml" / "splits" / "external_pandit.csv"
SEED = 42
TEST_DRAWN = 30  # patients tirés à l'étape 11b-7 (dont celui écarté)
DEV_COUNT = 40


def eligible_patients(patients: dict[str, dict]) -> list[str]:
    """Liste les patients qui ont T1, T2 ET masque (ordre numérique).

    Args:
        patients: Sortie de ``download_pandit.group_patients``.

    Returns:
        Identifiants triés.
    """
    return sorted((p for p, f in patients.items() if {"t1", "t2", "tumor"} <= set(f)), key=int)


def draw_dev_patients(
    eligible: list[str], test_drawn: list[str], count: int = DEV_COUNT, seed: int = SEED
) -> tuple[list[str], list[str]]:
    """Tire les patients de développement, disjoints de ceux du test, plus une liste de réserve.

    Args:
        eligible: Candidats (T1, T2 et masque), triés.
        test_drawn: Les patients tirés pour le test externe (à exclure).
        count: Nombre de patients de développement voulu.
        seed: Graine du tirage.

    Returns:
        ``(patients de développement triés, réserve dans l'ordre d'appel)``.

    Raises:
        ValueError: Si un patient tiré appartient au test.
    """
    pool = [p for p in eligible if p not in set(test_drawn)]
    chosen = random.Random(seed).sample(pool, min(count, len(pool)))
    rest = [p for p in pool if p not in set(chosen)]
    reserve = random.Random(seed).sample(rest, len(rest))
    if set(chosen) & set(test_drawn) or set(reserve) & set(test_drawn):
        raise ValueError("Un patient de développement appartient au test.")
    return sorted(chosen, key=int), reserve


def main() -> None:
    """Point d'entrée : tire les patients, extrait les coupes et écrit le manifeste."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--patients", type=int, default=DEV_COUNT, help="patients voulus")
    parser.add_argument("--min-free-gb", type=float, default=2.0, help="place libre minimale")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    # Les fonctions d'extraction du test écrivent dans des dossiers globaux : on les redirige.
    ex.SLICES_DIR = DEV_DIR / "slices"
    ex.OVERLAY_DIR = DEV_DIR / "overlay"
    remote = dp.HttpRangeFile(dp.FILE_URL, dp.ZIP_SIZE)
    rows: list[dict] = []
    kept: list[str] = []
    with zipfile.ZipFile(remote) as archive:
        patients = dp.group_patients(archive.infolist())
        eligible = eligible_patients(patients)
        test_drawn = dp.choose_patients(patients, TEST_DRAWN)
        with open(TEST_MANIFEST, encoding="utf-8", newline="") as file:
            in_manifest = {r["patient"] for r in csv.DictReader(file)}
        if not in_manifest <= set(test_drawn) or len(test_drawn) - len(in_manifest) != 1:
            raise SystemExit("Le tirage du test ne se reproduit pas : arrêt.")
        LOGGER.info(
            "%d candidats ; test reproduit (%d tirés, %d dans le manifeste, écarté : %s)",
            len(eligible),
            len(test_drawn),
            len(in_manifest),
            sorted(set(test_drawn) - in_manifest, key=int),
        )
        chosen, reserve = draw_dev_patients(eligible, test_drawn, args.patients)
        queue = chosen + reserve  # les premiers sont le tirage, la réserve remplace les écartés
        for pid in queue:
            if len(kept) >= args.patients:
                break
            if ex.dp.free_gb(ROOT) < args.min_free_gb + 0.5:
                raise SystemExit("Place libre insuffisante : arrêt.")
            paths = dp.fetch_patient(archive, patients[pid], pid, VOLUMES_DIR)
            try:
                new_rows = ex.process_patient(pid, paths)
            finally:
                for path in paths.values():
                    path.unlink(missing_ok=True)  # volumes supprimés après chaque patient
            if {r["sequence"] for r in new_rows} == {"T1", "T2"}:
                rows += new_rows
                kept.append(pid)
            else:
                LOGGER.warning("Patient %s écarté (T1 et T2 non disponibles ensemble)", pid)
                for r in new_rows:
                    (ex.SLICES_DIR / r["file"]).unlink(missing_ok=True)
            LOGGER.info("Patient %s traité (%d retenus sur %d)", pid, len(kept), args.patients)
    if set(kept) & set(test_drawn):
        raise SystemExit("Patient commun avec le test : arrêt.")
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST, "w", encoding="utf-8", newline="\n") as file:
        writer = csv.DictWriter(file, fieldnames=ex.COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    if VOLUMES_DIR.is_dir() and not any(VOLUMES_DIR.iterdir()):
        shutil.rmtree(VOLUMES_DIR)
    LOGGER.info("%d patients, %d PNG, manifeste %s", len(kept), len(rows), MANIFEST)


if __name__ == "__main__":
    main()
