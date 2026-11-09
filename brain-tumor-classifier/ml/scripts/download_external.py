"""Télécharge un jeu externe (PMRAM ou BDNeuro-MRI) dans ``data/external/<jeu>/``.

Jeux disponibles (option ``--dataset``) :
    - ``pmram`` (défaut) : « PMRAM: Bangladeshi Brain Cancer - MRI Dataset » (Mendeley Data),
      CC BY 4.0, https://data.mendeley.com/datasets/m7w55sw88b/1 ;
    - ``bdneuro`` : « BDNeuro-MRI » version 7 (Mendeley Data), CC BY 4.0,
      https://data.mendeley.com/datasets/zwr4ntf94j/7 (dossiers ``train/val/test``).
Ils servent de jeux de TEST uniquement (autres hôpitaux que Kaggle). Rien n'est écrit dans
git : le dossier ``data/`` est ignoré.

Étapes :
    1. liste les fichiers via l'API publique de Mendeley et télécharge l'archive d'images
       (pour PMRAM jamais ``Augmented Data.zip``), en vérifiant son SHA-256 contre celui
       de l'API ;
    2. ne garde que les images, rangées par classe (glioma, meningioma, notumor, pituitary) ;
    3. PMRAM seulement : écarte les doublons exacts (SHA-256) et quasi-doublons (dHash) avec
       tout Kaggle (``Training`` + ``Testing``) : les fichiers sont DÉPLACÉS dans
       ``_excluded/`` et listés dans ``duplicates_report.json``, rien n'est effacé.
       Pour BDNeuro, tout est gardé : le recouvrement avec Kaggle est mesuré ensuite par
       ``scripts/check_overlap.py`` (embeddings), qui met le manifeste à jour ;
    4. écrit ``manifest.json`` (fichier, classe, taille, mode, sha256, statut) ;
    5. génère ``contact_sheet.png`` pour un contrôle à l'œil.

Si le téléchargement automatique est impossible, téléchargez l'archive à la main sur la
page du jeu et posez-la dans ``data/external/<jeu>/_download/`` : le script la traite
telle quelle (ou indiquez-la avec ``--archive``).

Utilisation (depuis ``brain-tumor-classifier/ml/``) ::

    uv run python scripts/download_external.py [--dataset bdneuro] [--force]
        [--archive CHEMIN] [--threshold 6]
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import requests
import truststore
from PIL import Image, ImageDraw

from brain_mri.duplicates import IMAGE_SUFFIXES, TEST_DIR_NAME, TRAIN_DIR_NAME, sha256_file
from brain_mri.near_duplicates import (
    DEFAULT_THRESHOLD,
    Match,
    find_exact,
    find_near,
    fingerprint_folders,
)

# Réseau qui re-signe le HTTPS : on utilise le magasin de certificats du système
# (la vérification des certificats reste activée), comme download_data.py.
truststore.inject_into_ssl()

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # brain-tumor-classifier/
KAGGLE_DIR = PROJECT_ROOT / "data" / "raw"
DOWNLOAD_DIR_NAME = "_download"  # archives téléchargées (avec leur hash)
EXCLUDED_DIR_NAME = "_excluded"  # doublons déplacés ici
CLASS_NAMES = ("glioma", "meningioma", "notumor", "pituitary")  # noms de classes du projet
SPLIT_NAMES = ("train", "val", "test")  # découpages éventuels dans l'archive (BDNeuro)
SEED = 42  # graine du tirage de la planche
PER_CLASS = 8  # images par classe sur la planche
THUMB = 160  # côté d'une vignette, en pixels
HTTP_HEADERS = {"User-Agent": "Mozilla/5.0"}  # l'API refuse les requêtes sans User-Agent


@dataclass(frozen=True)
class DatasetSpec:
    """Description d'un jeu externe : où le prendre et comment lire son archive."""

    key: str  # nom court (dossier ``data/external/<key>/`` et option ``--dataset``)
    title: str  # titre complet du jeu
    dataset_id: str  # identifiant Mendeley
    version: int  # version du jeu sur Mendeley
    archive_name: str  # nom de l'archive d'images (la seule utilisée)
    folder_to_class: dict[str, str]  # dossier de classe de l'archive -> classe du projet
    dedupe: bool  # écarter dès le téléchargement les doublons de Kaggle (dHash) ?

    @property
    def api_files(self) -> str:
        """Adresse de l'API publique listant les fichiers du jeu."""
        return f"https://data.mendeley.com/public-api/datasets/{self.dataset_id}/files"

    @property
    def page(self) -> str:
        """Adresse de la page publique du jeu (bonne version)."""
        return f"https://data.mendeley.com/datasets/{self.dataset_id}/{self.version}"

    @property
    def directory(self) -> Path:
        """Dossier local du jeu : ``data/external/<key>/``."""
        return PROJECT_ROOT / "data" / "external" / self.key


# Dossiers de l'archive -> classes du projet (comparaison en minuscules, sans préfixe « 512 »).
SPECS = {
    "pmram": DatasetSpec(
        key="pmram",
        title="PMRAM: Bangladeshi Brain Cancer - MRI Dataset",
        dataset_id="m7w55sw88b",
        version=1,
        archive_name="Raw Data.zip",  # l'autre archive contient des images augmentées
        folder_to_class={
            "glioma": "glioma",
            "meningioma": "meningioma",
            "normal": "notumor",
            "pituitary": "pituitary",
        },
        dedupe=True,
    ),
    "bdneuro": DatasetSpec(
        key="bdneuro",
        title="BDNeuro-MRI: A Bangladeshi Clinical Brain Tumor MRI Dataset (v7)",
        dataset_id="zwr4ntf94j",
        version=7,
        archive_name="Brain_Tumor_MRI_Dataset_Final.zip",
        folder_to_class={
            "glioma": "glioma",
            "meningioma": "meningioma",
            "no_tumor": "notumor",
            "pituitary": "pituitary",
        },
        dedupe=False,
    ),
}


def sanitize_name(name: str) -> str:
    """Remplace les espaces et parenthèses d'un nom de fichier par des tirets bas."""
    return "".join("_" if c in " ()" else c for c in name).replace("__", "_").replace("_.", ".")


def class_of(member: str, spec: DatasetSpec) -> str | None:
    """Donne la classe du projet d'un chemin du zip, ou ``None`` s'il faut l'ignorer.

    Args:
        member: Chemin dans l'archive, par ex. ``Raw/512Glioma/glioma (59).jpg``.
        spec: Description du jeu (table dossier -> classe).

    Returns:
        La classe, ou ``None`` pour un dossier inconnu ou « augmenté » (jamais gardé).
    """
    parts = PurePosixPath(member).parts
    if len(parts) < 2 or any("augment" in p.lower() for p in parts):
        return None
    return spec.folder_to_class.get(parts[-2].lower().removeprefix("512"))


def fetch_archive(spec: DatasetSpec, download_dir: Path) -> tuple[Path, dict]:
    """Télécharge l'archive d'images depuis Mendeley et vérifie son SHA-256.

    Args:
        spec: Description du jeu.
        download_dir: Dossier où poser l'archive.

    Returns:
        Le chemin de l'archive et les métadonnées de l'API (hash annoncé, taille, id).

    Raises:
        RuntimeError: Si le fichier est introuvable côté API ou si le hash ne correspond pas.
    """
    listing = requests.get(
        spec.api_files,
        params={"folder_id": "root", "version": spec.version},
        headers=HTTP_HEADERS,
        timeout=60,
    )
    listing.raise_for_status()
    entry = next((f for f in listing.json() if f["filename"] == spec.archive_name), None)
    if entry is None:
        raise RuntimeError(f"{spec.archive_name} absent de la liste Mendeley")
    details = entry["content_details"]
    target = download_dir / spec.archive_name
    download_dir.mkdir(parents=True, exist_ok=True)
    if not (target.is_file() and sha256_file(target) == details["sha256_hash"]):
        print(f"Téléchargement de {spec.archive_name} ({details['size'] / 1e6:.1f} Mo) ...")
        with requests.get(
            details["download_url"], headers=HTTP_HEADERS, stream=True, timeout=120
        ) as response:
            response.raise_for_status()
            with open(target, "wb") as file:
                for chunk in response.iter_content(1 << 20):
                    file.write(chunk)
    if sha256_file(target) != details["sha256_hash"]:
        raise RuntimeError(f"SHA-256 de {target} différent de celui annoncé par Mendeley")
    return target, {"api_sha256": details["sha256_hash"], "size": details["size"]}


def extract_raw_images(archive: Path, root_dir: Path, spec: DatasetSpec) -> list[dict]:
    """Extrait uniquement les images, rangées par classe.

    Args:
        archive: Archive d'images du jeu.
        root_dir: Dossier de destination (``<classe>/<fichier>``).
        spec: Description du jeu.

    Returns:
        Une entrée de manifeste par image (sans le statut de doublon).
    """
    entries = []
    with zipfile.ZipFile(archive) as zf:
        for info in sorted(zf.infolist(), key=lambda i: i.filename):
            label = class_of(info.filename, spec)
            parts = PurePosixPath(info.filename).parts
            suffix = PurePosixPath(info.filename).suffix.lower()
            if info.is_dir() or label is None or suffix not in IMAGE_SUFFIXES:
                continue
            # On ne garde que le nom de base : aucun chemin du zip n'est réutilisé tel quel.
            name = sanitize_name(parts[-1])
            target = root_dir / label / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
            with Image.open(target) as image:
                width, height, mode = *image.size, image.mode
            entries.append(
                {
                    "file": f"{label}/{name}",
                    "class": label,
                    "split": parts[-3] if len(parts) >= 3 and parts[-3] in SPLIT_NAMES else None,
                    "source_member": info.filename,
                    "size_bytes": target.stat().st_size,
                    "width": width,
                    "height": height,
                    "mode": mode,
                    "sha256": sha256_file(target),
                    "status": "kept",
                }
            )
    return entries


def match_to_dict(match: Match, label: str, prefix: str) -> dict:
    """Convertit une correspondance en dictionnaire JSON (avec la classe du jeu externe)."""
    return {
        "file": match.external.removeprefix(f"{prefix}/"),
        "class": label,
        "kaggle_file": match.reference,
        "distance": match.distance,
    }


def exclude_duplicates(entries: list[dict], spec: DatasetSpec, threshold: int) -> dict:
    """Écarte du jeu externe les doublons exacts et quasi-doublons de Kaggle.

    Les fichiers sont déplacés dans ``_excluded/<classe>/`` et leur statut passe à
    ``excluded_exact`` ou ``excluded_near`` dans ``entries`` (modifié sur place).

    Args:
        entries: Manifeste des images extraites.
        spec: Description du jeu.
        threshold: Distance de Hamming maximale pour un quasi-doublon.

    Returns:
        Le rapport (seuil, nombres par classe, listes des correspondances).
    """
    root_dir, prefix = spec.directory, spec.key
    print("Empreintes de Kaggle (SHA-256 + dHash), cela peut prendre une minute ...")
    kaggle = fingerprint_folders([KAGGLE_DIR / TRAIN_DIR_NAME, KAGGLE_DIR / TEST_DIR_NAME])
    external = fingerprint_folders([root_dir / c for c in CLASS_NAMES if (root_dir / c).is_dir()])
    # fingerprint_folders préfixe par le nom du dossier de classe : on le remplace par le jeu.
    external = {f"{prefix}/{k}": v for k, v in external.items()}
    exact = {m.external: m for m in find_exact(external, kaggle)}
    near = {m.external: m for m in find_near(external, kaggle, threshold)}

    by_file = {f"{prefix}/{e['file']}": e for e in entries}
    counts = {c: {"exact": 0, "near": 0, "kept": 0} for c in CLASS_NAMES}
    report: dict = {"threshold_bits": threshold, "exact": [], "near": []}
    for key, entry in by_file.items():
        label = entry["class"]
        if key in exact:
            status, bucket, match = "excluded_exact", "exact", exact[key]
        elif key in near:
            status, bucket, match = "excluded_near", "near", near[key]
        else:
            counts[label]["kept"] += 1
            continue
        counts[label][bucket] += 1
        entry["status"] = status
        entry["kaggle_match"] = match.reference
        entry["distance"] = match.distance
        report[bucket].append(match_to_dict(match, label, prefix))
        destination = root_dir / EXCLUDED_DIR_NAME / entry["file"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(root_dir / entry["file"]), destination)
    report["per_class"] = counts
    return report


def make_contact_sheet(entries: list[dict], root_dir: Path) -> Path:
    """Dessine la planche de contrôle : 8 images tirées au hasard par classe (graine 42).

    Chaque vignette porte le nom du fichier, sa taille et son mode.

    Args:
        entries: Manifeste (seules les images au statut ``kept`` sont tirées).
        root_dir: Dossier du jeu externe.

    Returns:
        Le chemin de ``contact_sheet.png``.
    """
    label_height = 42  # place pour 3 lignes de texte sous chaque vignette
    rng = random.Random(SEED)
    sheet = Image.new(
        "RGB", (PER_CLASS * THUMB, len(CLASS_NAMES) * (THUMB + label_height)), "white"
    )
    draw = ImageDraw.Draw(sheet)
    for row, label in enumerate(CLASS_NAMES):
        kept = [e for e in entries if e["class"] == label and e["status"] == "kept"]
        for col, entry in enumerate(rng.sample(kept, min(PER_CLASS, len(kept)))):
            x, y = col * THUMB, row * (THUMB + label_height)
            with Image.open(root_dir / entry["file"]) as image:
                sheet.paste(image.convert("RGB").resize((THUMB, THUMB)), (x, y))
            file_name = entry["file"].split("/")[-1][:24]
            text = f"{label}\n{entry['width']}x{entry['height']} {entry['mode']}\n{file_name}"
            draw.multiline_text((x + 2, y + THUMB + 2), text, fill="black")
    path = root_dir / "contact_sheet.png"
    sheet.save(path)
    return path


def main() -> None:
    """Orchestre le téléchargement, le tri, le contrôle des doublons et la planche."""
    parser = argparse.ArgumentParser(description="Télécharge et trie un jeu externe.")
    parser.add_argument("--dataset", choices=sorted(SPECS), default="pmram", help="jeu à préparer")
    parser.add_argument("--force", action="store_true", help="refait tout depuis l'archive")
    parser.add_argument("--archive", type=Path, help="archive d'images déposée à la main")
    parser.add_argument("--threshold", type=int, default=DEFAULT_THRESHOLD, help="bits dHash")
    args = parser.parse_args()
    spec = SPECS[args.dataset]
    root_dir = spec.directory

    manifest_path = root_dir / "manifest.json"
    if manifest_path.is_file() and not args.force:
        print(f"Déjà préparé dans {root_dir} (--force pour refaire).")
        return

    download_dir = root_dir / DOWNLOAD_DIR_NAME
    api_info: dict = {}
    # Archive déposée à la main (--archive) ou emplacement du téléchargement automatique.
    archive = args.archive or download_dir / spec.archive_name
    try:
        if not args.archive:
            archive, api_info = fetch_archive(spec, download_dir)
    except (requests.RequestException, RuntimeError) as error:
        if not archive.is_file():
            raise SystemExit(
                f"Téléchargement automatique impossible ({error}).\n"
                f"Téléchargez « {spec.archive_name} » sur {spec.page} "
                f"et posez-le dans {download_dir}, puis relancez le script."
            ) from error
        print(f"Téléchargement impossible ({error}) : archive locale utilisée.")

    # Repart d'un état propre (ces dossiers sont générés par ce script).
    for name in (*CLASS_NAMES, EXCLUDED_DIR_NAME):
        shutil.rmtree(root_dir / name, ignore_errors=True)
    entries = extract_raw_images(archive, root_dir, spec)
    report: dict | None = None
    if spec.dedupe:
        report = exclude_duplicates(entries, spec, args.threshold)
        (root_dir / "duplicates_report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    manifest = {
        "dataset": spec.title,
        "source": spec.page,
        "licence": "CC BY 4.0",
        "archives": [
            {"file": archive.name, "sha256": sha256_file(archive), **api_info},
        ],
        "images": entries,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    sheet = make_contact_sheet(entries, root_dir)

    if report is not None:
        print(f"\n{'classe':<11} {'brutes':>7} {'exacts':>7} {'quasi':>6} {'gardées':>8}")
        for label, c in report["per_class"].items():
            total = c["exact"] + c["near"] + c["kept"]
            print(f"{label:<11} {total:>7} {c['exact']:>7} {c['near']:>6} {c['kept']:>8}")
    else:
        for label in CLASS_NAMES:
            print(f"{label:<11} {sum(e['class'] == label for e in entries):>7} images")
    print(f"\nManifeste : {manifest_path}\nPlanche : {sheet}")


if __name__ == "__main__":
    main()
