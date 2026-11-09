"""Télécharge le jeu figshare de Cheng (copie Hugging Face ``chehablab/JCBTD``), étape 11b-1.

Pourquoi : le jeu Kaggle fusionne plusieurs sources, dont ce jeu figshare (3 064 coupes IRM T1
avec contraste, 233 patients). Kaggle ne donne AUCUN identifiant de patient ; le jeu figshare
en donne un pour chaque coupe. On s'en sert pour CALIBRER la règle « deux images du même
patient » (voir ``calibrate_patient_threshold.py``) et pour contrôler le découpage
(``make_split.py``). Ce jeu n'est PAS utilisé pour entraîner ni pour tester.

Source : https://huggingface.co/datasets/chehablab/JCBTD, licence CC BY 4.0 (copie du jeu
figshare de Jun Cheng, doi 10.6084/m9.figshare.1512427). Étiquettes de la carte du jeu :
``1`` = meningioma, ``2`` = glioma, ``3`` = pituitary.

Étapes :
    1. utilise la révision épinglée ``REVISION`` (hash de commit du dépôt, vérifié le
       2026-11-09) ; ``--revision`` permet d'en imposer une autre ;
    2. télécharge le fichier parquet à cette révision ;
    3. contrôle le contenu (colonnes, 3 064 images, 233 patients, étiquettes) : s'arrête si
       un chiffre diffère ;
    4. extrait chaque image en PNG (tournée de 270 degrés, orientation de Kaggle) dans
       ``data/external/figshare_pid/images/`` et écrit ``labels.csv``
       (fichier, label, patient_id, sha256) et ``manifest.json``.

Rien n'est écrit dans git : ``data/`` est ignoré.

Utilisation (depuis ``brain-tumor-classifier/ml/``) ::

    uv run python scripts/download_figshare_pid.py [--revision HASH] [--force]
"""

from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path

import pyarrow.parquet as pq
import truststore
from huggingface_hub import hf_hub_download
from PIL import Image

from brain_mri.duplicates import sha256_file

# Réseau qui re-signe le HTTPS : magasin de certificats du système (vérification conservée)
truststore.inject_into_ssl()

REPO_ID = "chehablab/JCBTD"  # dépôt Hugging Face (type « dataset »)
REVISION = "2676e40a379f8cfc283d9fffea1f18275bee6924"  # commit épinglé : même contenu à chaque fois
PARQUET_FILE = "data/train-00000-of-00001.parquet"  # unique fichier de données du dépôt
LICENSE = "CC BY 4.0"  # licence lue dans la carte du jeu
LABEL_NAMES = {1: "meningioma", 2: "glioma", 3: "pituitary"}  # carte du jeu : 1, 2, 3
EXPECTED_IMAGES = 3064  # nombre de coupes annoncé par la carte
EXPECTED_PATIENTS = 233  # nombre de patients annoncé par la carte
EXPECTED_COLUMNS = {"patient_id", "slice_id", "image", "mask", "label"}
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # brain-tumor-classifier/
OUTPUT_DIR = PROJECT_ROOT / "data" / "external" / "figshare_pid"


def pinned_revision(requested: str | None) -> str:
    """Donne la révision (hash de commit) à télécharger.

    Args:
        requested: Hash imposé par l'utilisateur, ou ``None`` pour la révision épinglée.

    Returns:
        Le hash du commit à télécharger.
    """
    return requested or REVISION


def check_content(table) -> dict:
    """Contrôle que le contenu du parquet est celui annoncé par la carte du jeu.

    Args:
        table: Table pyarrow lue depuis le parquet.

    Returns:
        Un résumé (colonnes, effectifs, nombre de patients, étiquettes).

    Raises:
        ValueError: Si les colonnes, le nombre d'images, le nombre de patients ou les
            étiquettes diffèrent de la carte.
    """
    columns = set(table.column_names)
    if columns != EXPECTED_COLUMNS:
        raise ValueError(f"Colonnes inattendues : {sorted(columns)}")
    labels = table.column("label").to_pylist()
    patients = set(table.column("patient_id").to_pylist())
    if len(labels) != EXPECTED_IMAGES:
        raise ValueError(f"{len(labels)} images au lieu de {EXPECTED_IMAGES}")
    if len(patients) != EXPECTED_PATIENTS:
        raise ValueError(f"{len(patients)} patients au lieu de {EXPECTED_PATIENTS}")
    if not set(labels) <= set(LABEL_NAMES):
        raise ValueError(f"Étiquettes inattendues : {sorted(set(labels))}")
    return {
        "columns": table.column_names,
        "n_images": len(labels),
        "n_patients": len(patients),
        "labels": {LABEL_NAMES[k]: labels.count(k) for k in sorted(LABEL_NAMES)},
    }


def extract(table, output_dir: Path) -> list[dict]:
    """Écrit chaque image en PNG et renvoie la liste des lignes du CSV.

    Args:
        table: Table pyarrow du jeu.
        output_dir: Dossier du jeu (les PNG vont dans ``images/``).

    Returns:
        Une ligne par image : ``file`` (relatif à ``output_dir``), ``label`` (nom de classe),
        ``patient_id``, ``sha256``, ``width``, ``height``, ``mode``.
    """
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for record in table.to_pylist():
        label = LABEL_NAMES[record["label"]]
        name = f"{label}_{int(record['slice_id']):04d}.png"
        path = images_dir / name
        raw = record["image"]["bytes"]
        with Image.open(io.BytesIO(raw)) as image:
            width, height = image.size
            mode = image.mode
            # Orientation de Kaggle (constatée : corrélation 1,0 sur des miniatures) : les
            # coupes figshare y sont tournées de 270 degrés (sens anti-horaire) par rapport au
            # fichier d'origine. Sans cela, le cosinus avec Kaggle reste sous 0,97.
            image.transpose(Image.Transpose.ROTATE_270).save(path, format="PNG")
        rows.append(
            {
                "file": f"images/{name}",
                "label": label,
                "patient_id": str(record["patient_id"]),
                "sha256": sha256_file(path),
                "width": width,
                "height": height,
                "mode": mode,
            }
        )
    return rows


def main() -> None:
    """Point d'entrée : télécharge, contrôle, extrait et affiche le contenu réel."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--revision", help="hash de commit à télécharger (défaut : REVISION)")
    parser.add_argument(
        "--force", action="store_true", help="refait l'extraction même si elle existe"
    )
    args = parser.parse_args()

    if (OUTPUT_DIR / "labels.csv").is_file() and not args.force:
        print(f"Déjà extrait : {OUTPUT_DIR} (utilisez --force pour refaire)")
        return
    revision = pinned_revision(args.revision)
    print(f"Révision épinglée : {revision}")
    parquet_path = hf_hub_download(REPO_ID, PARQUET_FILE, repo_type="dataset", revision=revision)
    table = pq.read_table(parquet_path)
    print(f"Colonnes : {table.column_names} ; lignes : {table.num_rows}")
    summary = check_content(table)
    print(f"Contrôle OK : {summary['n_images']} images, {summary['n_patients']} patients")
    print(f"Étiquettes (1=meningioma, 2=glioma, 3=pituitary) : {summary['labels']}")

    rows = extract(table, OUTPUT_DIR)
    with open(OUTPUT_DIR / "labels.csv", "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    sizes: dict[str, int] = {}
    for row in rows:
        key = f"{row['width']}x{row['height']} {row['mode']}"
        sizes[key] = sizes.get(key, 0) + 1
    manifest = {
        "repo": REPO_ID,
        "revision": revision,
        "license": LICENSE,
        "source": "https://huggingface.co/datasets/chehablab/JCBTD",
        "orientation": "images tournées de 270 degrés (ROTATE_270) pour retrouver celle de Kaggle",
        "label_names": {str(k): v for k, v in LABEL_NAMES.items()},
        "image_sizes": sizes,
        **summary,
    }
    (OUTPUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Tailles et modes des images : {sizes}")
    print(f"{len(rows)} PNG et labels.csv écrits dans {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
