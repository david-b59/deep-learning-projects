"""Télécharge le jeu « hors sujet » des garde-fous : 100 images par catégorie (étape 11c-0).

But : mesurer combien d'images qui ne sont PAS des IRM cérébrales les garde-fous refusent
(critère G2 de ``ml/splits/guards_rules.md``). Ce jeu sert uniquement à MESURER, jamais à
ajuster un seuil. Graine 42, 100 images par catégorie, dans ``data/external/offtopic/``
(ignoré par git). Manifeste versionné ``ml/splits/offtopic.csv`` (aucune image) :
``category, source, license, url, sha256, file``.

Catégories et sources (licences vérifiées le 2026-11-11) :

* ``ultrasound``, ``chest_xray``, ``fundus``, ``abdominal_ct`` : MedMNIST+ (Zenodo, enregistrement
  10519652, CC BY 4.0) : BreastMNIST 224, PneumoniaMNIST 224, RetinaMNIST 224, OrganAMNIST 128.
  Les jeux de ce lot sont sous CC BY 4.0, SAUF DermaMNIST (CC BY-NC) qui n'est JAMAIS utilisé.
  Les fichiers ``.npz`` sont chargés avec ``allow_pickle=False`` ; le plus gros
  (OrganAMNIST 128, 708 Mo) est téléchargé dans un dossier temporaire, échantillonné puis supprimé.
* ``photo`` : Wikimedia Commons via l'API MediaWiki ; la licence est lue image par image dans
  ``extmetadata`` et seules CC0, domaine public et CC BY sont gardées (pas de « SA », pas de NC) ;
  vignettes de 512 px de large.
* ``spine_mri`` : IRM lombaires (jeu « LSMA-PQR », Mendeley Data p3r4xd2488, CC BY 4.0, dérivé de
  Lumbar Spine MRI k57fr854j2, lui aussi CC BY 4.0). Le jeu d'origine (6,3 Go en DICOM) est trop
  gros ; on télécharge les 100 seules images PNG tirées dans le jeu dérivé (de l'ordre de 300 Ko
  chacune), pas l'archive.

Utilisation (depuis ``brain-tumor-classifier/ml/``) ::

    PYTHONUTF8=1 uv run python scripts/download_offtopic.py [--only photo spine_mri ...]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import logging
import random
import re
import shutil
import time
from pathlib import Path

import numpy as np
import requests
from PIL import Image

try:  # réseau qui re-signe le HTTPS : magasin de certificats du système, si disponible
    import truststore

    truststore.inject_into_ssl()
except ImportError:  # pragma: no cover - dépend de l'environnement
    pass

LOGGER = logging.getLogger("download_offtopic")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "data" / "external" / "offtopic"
TMP_DIR = OUTPUT_DIR / "_tmp"
MANIFEST = PROJECT_ROOT / "ml" / "splits" / "offtopic.csv"
COLUMNS = ["category", "source", "license", "url", "sha256", "file"]
SEED = 42
PER_CATEGORY = 100
USER_AGENT = "brain-tumor-classifier-offtopic/1.0 (projet pedagogique, mesure hors ligne)"
ZENODO_RECORD = "https://zenodo.org/api/records/10519652"
ZENODO_LICENSE = "CC BY 4.0"
MEDMNIST = {  # catégorie -> (fichier npz, jeu d'origine)
    "ultrasound": ("breastmnist_224.npz", "BreastMNIST 224"),
    "chest_xray": ("pneumoniamnist_224.npz", "PneumoniaMNIST 224"),
    "fundus": ("retinamnist_224.npz", "RetinaMNIST 224"),
    "abdominal_ct": ("organamnist_128.npz", "OrganAMNIST 128"),
}
FORBIDDEN_SUBSETS = ("dermamnist",)  # CC BY-NC : interdit
WIKI_API = "https://commons.wikimedia.org/w/api.php"
WIKI_QUERIES = [  # jusqu'à 10 photos par sujet, 100 au total (aucun lien avec le cerveau)
    "cat", "dog", "mountain landscape", "city street", "car", "flower",
    "beach", "food dish", "bridge", "bicycle", "tree", "house", "boat", "bird",
]  # fmt: skip
WIKI_ALLOWED = re.compile(
    r"^(CC0|Public domain|PD\b|CC BY(?: |-)\d(?:\.\d)?(?: [A-Za-z]{2,3})?$)", re.I
)
MENDELEY_API = "https://data.mendeley.com/public-api/datasets/p3r4xd2488"
MENDELEY_IMAGE_FOLDER = "4319d226-fc53-4076-8074-c0d975777386"  # dossier des images (RGB 384x384)
SPINE_NAME = re.compile(r"(axial|sag)_T[12]_\d+_(?:D|S)\d+\.png")


def sample_indices(total: int, count: int, seed: int = SEED) -> list[int]:
    """Tire ``count`` indices distincts parmi ``total`` (graine fixe, résultat trié).

    Args:
        total: Taille de la population.
        count: Nombre d'indices voulu (plafonné à ``total``).
        seed: Graine du tirage.

    Returns:
        Indices triés par ordre croissant.
    """
    return sorted(random.Random(seed).sample(range(total), min(count, total)))


def is_allowed_license(short_name: str) -> bool:
    """Dit si une licence Wikimedia est acceptée : CC0, domaine public ou CC BY (sans SA/NC/ND).

    Args:
        short_name: Valeur ``LicenseShortName`` de ``extmetadata`` (ex. ``CC BY 4.0``).

    Returns:
        True si la licence est dans la liste autorisée.
    """
    return bool(WIKI_ALLOWED.match(short_name.strip()))


def sha256_bytes(data: bytes) -> str:
    """Empreinte SHA-256 (hexadécimale) d'octets."""
    return hashlib.sha256(data).hexdigest()


def save_png(array_or_image: np.ndarray | Image.Image, path: Path) -> str:
    """Enregistre une image en PNG et renvoie son SHA-256.

    Args:
        array_or_image: Tableau ``uint8`` (H, W) ou (H, W, 3), ou image PIL.
        path: Fichier de sortie (dossier créé au besoin).

    Returns:
        L'empreinte SHA-256 du fichier écrit.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    image = (
        array_or_image
        if isinstance(array_or_image, Image.Image)
        else Image.fromarray(array_or_image)
    )
    image.save(path)
    return sha256_bytes(path.read_bytes())


def download_file(url: str, target: Path, md5: str | None = None) -> None:
    """Télécharge un gros fichier en flux et contrôle son MD5.

    Args:
        url: Adresse du fichier.
        target: Fichier local de destination.
        md5: Somme MD5 attendue (``None`` : pas de contrôle).

    Raises:
        ValueError: Si la somme MD5 ne correspond pas.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.md5()  # noqa: S324 - contrôle d'intégrité fourni par Zenodo, pas de sécurité
    with requests.get(url, stream=True, timeout=120, headers={"User-Agent": USER_AGENT}) as r:
        r.raise_for_status()
        with open(target, "wb") as file:
            for chunk in r.iter_content(1 << 20):
                file.write(chunk)
                digest.update(chunk)
    if md5 is not None and digest.hexdigest() != md5:
        raise ValueError(f"Somme MD5 inattendue pour {target.name}")


def fetch_medmnist(category: str) -> list[dict]:
    """Télécharge un jeu MedMNIST+, tire 100 images du lot de test et les enregistre en PNG.

    Args:
        category: Une clé de ``MEDMNIST``.

    Returns:
        Lignes du manifeste.
    """
    name, label = MEDMNIST[category]
    if name.split("_")[0] in FORBIDDEN_SUBSETS:
        raise ValueError("Jeu interdit (licence non commerciale).")
    record = requests.get(ZENODO_RECORD, timeout=60, headers={"User-Agent": USER_AGENT}).json()
    if record["metadata"]["license"]["id"] != "cc-by-4.0":
        raise SystemExit("La licence de l'enregistrement Zenodo n'est plus CC BY 4.0 : arrêt.")
    entry = next(f for f in record["files"] if f["key"] == name)
    npz_path = TMP_DIR / name
    LOGGER.info("%s : téléchargement de %s (%.0f Mo)", category, name, entry["size"] / 1e6)
    download_file(entry["links"]["self"], npz_path, entry["checksum"].split(":", 1)[1])
    rows = []
    try:
        with np.load(npz_path, allow_pickle=False) as data:  # tableaux numpy seulement
            images = data["test_images"]
        for k in sample_indices(len(images), PER_CATEGORY):
            file = f"{category}_{k:05d}.png"
            sha = save_png(images[k], OUTPUT_DIR / category / file)
            rows.append(
                {
                    "category": category,
                    "source": f"MedMNIST+ {label} (Zenodo 10519652)",
                    "license": ZENODO_LICENSE,
                    "url": entry["links"]["self"],
                    "sha256": sha,
                    "file": f"{category}/{file}",
                }
            )
    finally:
        npz_path.unlink(missing_ok=True)  # le .npz temporaire est supprimé dans tous les cas
    return rows


def fetch_photos() -> list[dict]:
    """Télécharge 100 photos Wikimedia Commons sous licence libre (CC0, domaine public, CC BY).

    Pour chaque sujet de ``WIKI_QUERIES``, on cherche des fichiers JPEG, on lit la licence de
    chaque image dans ``extmetadata`` et on garde les 10 premières acceptées (ordre mélangé avec
    la graine 42). Vignettes de 512 px. Les résultats de la recherche Wikimedia peuvent changer
    avec le temps : c'est ``offtopic.csv`` (titres, URL, sha256) qui fait foi pour reproduire le jeu.

    Returns:
        Lignes du manifeste.
    """
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    rows: list[dict] = []
    per_query = 10  # au plus 10 images par sujet ; les derniers sujets complètent jusqu'à 100
    for query in WIKI_QUERIES:
        if len(rows) >= PER_CATEGORY:
            break
        params = {
            "action": "query",
            "format": "json",
            "generator": "search",
            "gsrsearch": f"{query} filetype:bitmap",
            "gsrnamespace": 6,
            "gsrlimit": 50,
            "prop": "imageinfo",
            "iiprop": "url|extmetadata|mime",
            "iiurlwidth": 512,
            "iiextmetadatafilter": "LicenseShortName",
        }
        pages = list(
            session.get(WIKI_API, params=params, timeout=60).json()["query"]["pages"].values()
        )
        random.Random(SEED).shuffle(pages)
        kept = 0
        for page in pages:
            info = (page.get("imageinfo") or [{}])[0]
            licence = info.get("extmetadata", {}).get("LicenseShortName", {}).get("value", "")
            if (
                kept >= per_query
                or len(rows) >= PER_CATEGORY
                or info.get("mime") != "image/jpeg"
                or not is_allowed_license(licence)
            ):
                continue
            time.sleep(0.5)  # politesse envers le serveur
            response = session.get(info["thumburl"], timeout=60)
            if response.status_code != 200:
                continue
            image = Image.open(io.BytesIO(response.content)).convert("RGB")
            file = f"photo_{len(rows):03d}.png"
            sha = save_png(image, OUTPUT_DIR / "photo" / file)
            rows.append(
                {
                    "category": "photo",
                    "source": f"Wikimedia Commons : {page['title']}",
                    "license": licence,
                    "url": info["descriptionurl"],
                    "sha256": sha,
                    "file": f"photo/{file}",
                }
            )
            kept += 1
        LOGGER.info("photo (%s) : %d images", query, kept)
    return rows


def fetch_spine() -> list[dict]:
    """Télécharge 100 images d'IRM lombaires (jeu dérivé Mendeley p3r4xd2488, CC BY 4.0).

    Returns:
        Lignes du manifeste.

    Raises:
        SystemExit: Si la licence n'est plus CC BY 4.0.
    """
    meta = requests.get(MENDELEY_API, timeout=60, headers={"User-Agent": USER_AGENT}).json()
    if meta["data_licence"]["short_name"] != "CC BY 4.0":
        raise SystemExit("Licence du jeu rachis différente de CC BY 4.0 : catégorie abandonnée.")
    candidates = sorted(
        (f for f in meta["files"]
         if f.get("folder_id") == MENDELEY_IMAGE_FOLDER and SPINE_NAME.fullmatch(f["filename"])),
        key=lambda f: f["filename"],
    )  # fmt: skip
    rows = []
    for k in sample_indices(len(candidates), PER_CATEGORY):
        entry = candidates[k]
        content = requests.get(
            entry["content_details"]["download_url"], timeout=60, headers={"User-Agent": USER_AGENT}
        ).content
        if sha256_bytes(content) != entry["content_details"]["sha256_hash"]:
            raise ValueError(f"Empreinte inattendue pour {entry['filename']}")
        file = f"spine_mri_{entry['filename']}"
        (OUTPUT_DIR / "spine_mri").mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / "spine_mri" / file).write_bytes(content)
        rows.append(
            {
                "category": "spine_mri",
                "source": f"LSMA-PQR (Mendeley Data p3r4xd2488) : {entry['filename']}",
                "license": "CC BY 4.0",
                "url": entry["content_details"]["download_url"],
                "sha256": sha256_bytes(content),
                "file": f"spine_mri/{file}",
            }
        )
    return rows


def main() -> None:
    """Point d'entrée : télécharge les catégories demandées et met à jour le manifeste."""
    categories = [*MEDMNIST, "photo", "spine_mri"]
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", nargs="+", choices=categories, help="catégories à (re)faire")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    todo = args.only or categories
    existing: list[dict] = []
    if MANIFEST.is_file():
        with open(MANIFEST, encoding="utf-8", newline="") as file:
            existing = [r for r in csv.DictReader(file) if r["category"] not in todo]
    rows = existing
    for category in todo:
        if category in MEDMNIST:
            rows += fetch_medmnist(category)
        elif category == "photo":
            rows += fetch_photos()
        else:
            rows += fetch_spine()
        LOGGER.info("%s : terminé", category)
    rows.sort(key=lambda r: (categories.index(r["category"]), r["file"]))
    with open(MANIFEST, "w", encoding="utf-8", newline="\n") as file:
        writer = csv.DictWriter(file, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    if TMP_DIR.is_dir():
        shutil.rmtree(TMP_DIR)
    LOGGER.info("%d images, manifeste %s", len(rows), MANIFEST)


if __name__ == "__main__":
    main()
