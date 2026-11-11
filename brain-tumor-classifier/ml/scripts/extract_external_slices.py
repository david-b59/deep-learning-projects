"""Extrait les coupes PNG du test externe Pandit 2025 et écrit le manifeste (étape 11b-7).

Règles : voir ``brain_mri.external_slices`` et ``ml/splits/external_rules.md`` (écrites avant
toute mesure). Pour chaque patient : coupe tumeur (masque maximal) + 2 à 3 coupes saines à
20 mm ou plus de la tumeur, mêmes indices pour T1 et T2 si le T2 est recalé sur la T1.

Deux façons d'obtenir les volumes :
    ``--fetch``   lit le zip distant par requêtes Range, patient par patient (voir
                  ``download_pandit.py``), extrait les coupes puis SUPPRIME les volumes
                  (T1 ~240 Mo + T2 ~160 Mo par patient : on garde donc moins de 1 Go) ;
    sans option   lit des volumes déjà présents dans ``data/external/pandit/volumes/``
                  (``N_t1.nii.gz``, ``N_t2.nii.gz``, ``N_tumor.nii.gz``).

Sorties : PNG dans ``data/external/pandit/slices/`` (ignoré par git), manifeste versionné
``ml/splits/external_pandit.csv`` (aucune image) et planche de contrôle
``data/cache/external_sheet.png``.

Utilisation (depuis ``brain-tumor-classifier/ml/``) ::

    PYTHONUTF8=1 uv run python scripts/extract_external_slices.py --fetch --patients 30
"""

from __future__ import annotations

import argparse
import csv
import logging
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))  # pour importer download_pandit

import download_pandit as dp  # noqa: E402

from brain_mri.duplicates import sha256_file  # noqa: E402
from brain_mri.external_slices import (  # noqa: E402
    HEALTHY,
    TUMOR,
    distance_to_mask_mm,
    load_volume,
    same_grid,
    save_png,
    select_healthy_slices,
    slice_to_uint8,
    tumor_slice_index,
    window_bounds,
)

LOGGER = logging.getLogger("extract_external_slices")
ROOT = dp.PROJECT_ROOT
DATA_DIR = dp.OUTPUT_DIR
SLICES_DIR = DATA_DIR / "slices"
VOLUMES_DIR = DATA_DIR / "volumes"
MANIFEST = ROOT / "ml" / "splits" / "external_pandit.csv"
OVERLAY_DIR = ROOT / "data" / "cache" / "external_overlay"
SHEET = ROOT / "data" / "cache" / "external_sheet.png"
COLUMNS = ["file", "patient", "sequence", "label", "slice_index", "distance_mm", "sha256"]
SEQUENCE_NAMES = {"t1": "T1", "t2": "T2"}  # rôle du zip -> nom de séquence du manifeste


def save_overlay(slice_xy: np.ndarray, mask_xy: np.ndarray, bounds: tuple, pid: str) -> None:
    """Enregistre la coupe tumeur avec le contour du masque en rouge (contrôle visuel).

    Args:
        slice_xy: Coupe T1 indexée ``[x, y]``.
        mask_xy: Masque de la même coupe.
        bounds: Fenêtre d'intensité.
        pid: Identifiant du patient.
    """
    gray = slice_to_uint8(slice_xy, bounds)
    edge = slice_to_uint8((mask_xy > 0).astype(np.float32), (0.0, 1.0)) > 0
    inner = (
        edge
        & np.roll(edge, 1, 0)
        & np.roll(edge, -1, 0)
        & np.roll(edge, 1, 1)
        & np.roll(edge, -1, 1)
    )
    rgb = np.stack([gray] * 3, axis=-1)
    rgb[edge & ~inner] = (255, 0, 0)
    OVERLAY_DIR.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(OVERLAY_DIR / f"p{int(pid):03d}_tumor_overlay.png")


def process_patient(pid: str, paths: dict[str, Path]) -> list[dict]:
    """Extrait les coupes d'un patient et renvoie ses lignes de manifeste.

    Args:
        pid: Identifiant du patient.
        paths: ``{rôle: fichier}`` avec ``t1``, ``tumor`` et éventuellement ``t2``.

    Returns:
        Lignes du manifeste (liste vide si le patient est écarté).
    """
    t1, zooms, t1_affine = load_volume(paths["t1"])
    mask, _, mask_affine = load_volume(paths["tumor"])
    if not same_grid(t1_affine, t1.shape, mask_affine, mask.shape):
        LOGGER.warning("Patient %s écarté : masque et T1 sur des grilles différentes", pid)
        return []
    t1_shape = t1.shape
    tumor_index = tumor_slice_index(mask)
    healthy = select_healthy_slices(mask, t1, zooms[2])
    if not healthy:
        LOGGER.warning("Patient %s écarté : moins de 2 coupes saines admissibles", pid)
        return []
    distances = distance_to_mask_mm(mask, zooms[2])
    chosen = [(TUMOR, tumor_index)] + [(HEALTHY, k) for k in healthy]
    rows = []
    # Un volume à la fois en mémoire (7,7 Go de RAM partagés) : T1 d'abord, puis T2.
    for role in ("t1", "t2"):
        if role == "t1":
            volume = t1
        elif "t2" in paths:
            volume, _, affine = load_volume(paths["t2"])
            if not same_grid(t1_affine, t1_shape, affine, volume.shape):
                LOGGER.warning("Patient %s : T2 non recalé sur la T1, T1 seule", pid)
                break
        else:
            break
        bounds = window_bounds(volume)  # une fenêtre par volume
        for label, k in chosen:
            name = f"p{int(pid):03d}_{role}_{label}_z{k:03d}.png"
            save_png(slice_to_uint8(volume[:, :, k], bounds), SLICES_DIR / name)
            rows.append(
                {
                    "file": name,
                    "patient": pid,
                    "sequence": SEQUENCE_NAMES[role],
                    "label": label,
                    "slice_index": k,
                    "distance_mm": f"{distances[k]:.1f}",
                    "sha256": sha256_file(SLICES_DIR / name),
                }
            )
        if role == "t1":  # contrôle visuel : contour du masque sur la coupe tumeur (hors git)
            save_overlay(volume[:, :, tumor_index], mask[:, :, tumor_index], bounds, pid)
            del t1  # libère le gros volume T1 avant de charger le T2
        del volume
    return rows


def make_sheet(rows: list[dict], path: Path, patients: int = 4) -> None:
    """Fabrique la planche de contrôle : une ligne par patient, T1 puis T2 (tumeur, saines).

    Args:
        rows: Lignes du manifeste.
        patients: Nombre de patients affichés.
        path: Fichier PNG de sortie.
    """
    pids = sorted({r["patient"] for r in rows}, key=int)[:patients]
    cell = 200
    sheet = Image.new("L", (cell * 8, cell * len(pids)), 0)
    draw = ImageDraw.Draw(sheet)
    for i, pid in enumerate(pids):
        mine = sorted(
            (r for r in rows if r["patient"] == pid),
            key=lambda r: (r["sequence"], r["label"] != TUMOR, int(r["slice_index"])),
        )
        for j, row in enumerate(mine[:8]):
            with Image.open(SLICES_DIR / row["file"]) as image:
                sheet.paste(image.resize((cell, cell)), (j * cell, i * cell))
            text = f"p{pid} {row['sequence']} {row['label']} z{row['slice_index']}"
            draw.text((j * cell + 4, i * cell + 4), text, fill=255)
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def main() -> None:
    """Point d'entrée : extrait les coupes, écrit le manifeste et la planche."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fetch", action="store_true", help="lire le zip distant (Range)")
    parser.add_argument("--patients", type=int, default=30, help="nombre de patients (30 max)")
    parser.add_argument("--min-free-gb", type=float, default=3.0, help="place libre minimale")
    parser.add_argument("--keep-volumes", action="store_true", help="ne pas supprimer les volumes")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if args.patients > 30:
        raise SystemExit("30 patients au maximum (règle de l'étape).")
    rows: list[dict] = []
    if args.fetch:
        remote = dp.HttpRangeFile(dp.FILE_URL, dp.ZIP_SIZE)
        with zipfile.ZipFile(remote) as archive:
            patients = dp.group_patients(archive.infolist())
            for pid in dp.choose_patients(patients, args.patients):
                if dp.free_gb(ROOT) < args.min_free_gb + 0.5:
                    raise SystemExit("Place libre insuffisante : arrêt.")
                paths = dp.fetch_patient(archive, patients[pid], pid, VOLUMES_DIR)
                rows += process_patient(pid, paths)
                LOGGER.info("Patient %s traité (%d lignes au total)", pid, len(rows))
                if not args.keep_volumes:
                    for path in paths.values():
                        path.unlink()
        LOGGER.info(
            "%d requêtes Range, %.0f Mo lus", remote.requests_count, remote.bytes_read / 1e6
        )
    else:
        pids = sorted({p.name.split("_")[0] for p in VOLUMES_DIR.glob("*_t1.nii.gz")}, key=int)
        for pid in pids[: args.patients]:
            paths = {r: VOLUMES_DIR / f"{pid}_{r}.nii.gz" for r in ("t1", "t2", "tumor")}
            rows += process_patient(pid, {r: p for r, p in paths.items() if p.is_file()})
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST, "w", encoding="utf-8", newline="\n") as file:
        writer = csv.DictWriter(file, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    make_sheet(rows, SHEET)
    kept = len({r["patient"] for r in rows})
    size_mb = sum(p.stat().st_size for p in SLICES_DIR.glob("*.png")) / 1e6
    LOGGER.info("%d patients, %d PNG (%.0f Mo), manifeste %s", kept, len(rows), size_mb, MANIFEST)
    if not args.keep_volumes and VOLUMES_DIR.is_dir() and not any(VOLUMES_DIR.iterdir()):
        shutil.rmtree(VOLUMES_DIR)  # dossier vide créé par ce script
    print(f"Planche de contrôle : {np.asarray(Image.open(SHEET)).shape} -> {SHEET}")


if __name__ == "__main__":
    main()
