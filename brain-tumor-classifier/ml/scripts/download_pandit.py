"""Télécharge un petit échantillon du jeu « Pandit 2025 » par requêtes HTTP Range (étape 11b-7).

Jeu : Pandit AS et coll., « Mapping Pituitary Neuroendocrine Tumors: An Annotated MRI Dataset
Profiling Tumor and Carotid Characteristics », Scientific Data, 2025,
doi 10.1038/s41597-024-04218-8. Données : figshare, doi 10.6084/m9.figshare.27894084.v1,
licence CC BY 4.0. Contenu vérifié (API figshare, répertoire central du zip, en-têtes NIfTI) :
136 patients ;
par patient, ``N_T1.nii.gz`` (T1 avec contraste, ~240 Mo, 0,5 mm isotrope, 136 patients),
``N_T2_f.nii.gz`` (T2 recalé sur la T1 : même grille et même affine, 100 patients),
``N_tumor_f.nii.gz`` (masque du complexe tumeur-glande hypophysaire, 135 patients, ~270 Ko) et
``N_carotids_f.nii.gz``. Pas de FLAIR. (Un patient a ``N_tumor.nii.gz`` sans ``_f``.)

Le jeu est un seul zip d'environ 47 Go. On ne le télécharge JAMAIS en entier : un objet fichier
« seekable » maison (:class:`HttpRangeFile`) est passé à :mod:`zipfile`, qui lit le répertoire
central (quelques centaines de Ko) puis seulement les octets des fichiers choisis, par requêtes
``Range``. Aucune dépendance de plus.

Ce script ne sert qu'à TESTER des modèles déjà entraînés : aucun entraînement.

Modes :
    ``--list``        lit le répertoire central et affiche le contenu (patients, tailles) ;
    ``--download``    télécharge les fichiers des N patients tirés au sort (graine 42) qui ont
                      T1, T2 et masque ; contrôle la place libre avant chaque fichier et
                      s'arrête sous ``--min-free-gb`` ou au-delà de ``--max-total-gb``.

Les volumes vont dans ``data/external/pandit/volumes/`` (ignoré par git). Une T1 pèse 240 Mo :
30 patients ne tiennent pas dans 1 Go. ``extract_external_slices.py --fetch`` utilise donc
:func:`fetch_patient` patient par patient, extrait les coupes PNG puis SUPPRIME les volumes.

Utilisation (depuis ``brain-tumor-classifier/ml/``) ::

    PYTHONUTF8=1 uv run python scripts/download_pandit.py --list
    PYTHONUTF8=1 uv run python scripts/download_pandit.py --download --patients 30
"""

from __future__ import annotations

import argparse
import io
import logging
import random
import re
import shutil
import zipfile
from pathlib import Path

import requests

try:  # réseau qui re-signe le HTTPS : magasin de certificats du système, si disponible
    import truststore

    truststore.inject_into_ssl()
except ImportError:  # pragma: no cover - dépend de l'environnement
    pass

LOGGER = logging.getLogger("download_pandit")
DOI_DATA = "10.6084/m9.figshare.27894084.v1"
DOI_PAPER = "10.1038/s41597-024-04218-8"
FILE_URL = "https://ndownloader.figshare.com/files/50793273"  # redirige vers une URL S3 signée
ZIP_SIZE = 47_245_902_155  # taille annoncée par l'API figshare (octets)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "data" / "external" / "pandit"
SEED = 42
BLOCK = 1 << 20  # lecture par blocs de 1 Mio (limite le nombre de requêtes)
NAME_PATTERN = re.compile(r"(?:.*/)?(\d+)_(T1|T2|tumou?r|carotids)(_f)?\.nii\.gz")
ROLES = {"T1": "t1", "T2": "t2", "tumor": "tumor", "tumour": "tumor"}  # carotides : ignorées


class HttpRangeFile(io.RawIOBase):
    """Fichier distant en lecture seule, « seekable », lu par requêtes HTTP ``Range``.

    Peut être passé à :class:`zipfile.ZipFile`. Un petit cache d'un bloc évite de refaire une
    requête pour des lectures voisines. L'URL signée (S3) est renouvelée si elle expire.
    """

    def __init__(self, url: str, size: int | None = None) -> None:
        """Résout l'URL signée et lit la taille du fichier.

        Args:
            url: Adresse de téléchargement (peut rediriger).
            size: Taille connue en octets ; sinon lue dans l'en-tête ``Content-Range``.
        """
        super().__init__()
        self.url = url
        self.session = requests.Session()
        self.position = 0
        self.cache_start = 0
        self.cache = b""
        self.requests_count = 0
        self.bytes_read = 0
        self.resolved = self._resolve()
        self.size = size if size is not None else self._probe_size()

    def _resolve(self) -> str:
        """Suit la redirection et renvoie l'URL finale (signée)."""
        with self.session.get(
            self.url, headers={"Range": "bytes=0-0"}, stream=True, timeout=60
        ) as response:
            response.raise_for_status()
            return response.url

    def _probe_size(self) -> int:
        """Lit la taille totale dans l'en-tête ``Content-Range``."""
        with self.session.get(
            self.resolved, headers={"Range": "bytes=0-0"}, stream=True, timeout=60
        ) as response:
            return int(response.headers["Content-Range"].split("/")[1])

    def _fetch(self, start: int, end: int) -> bytes:
        """Télécharge les octets ``start`` à ``end`` inclus (renouvelle l'URL si expirée).

        Args:
            start: Premier octet.
            end: Dernier octet (inclus).

        Returns:
            Les octets demandés.

        Raises:
            OSError: Si le serveur refuse la requête Range.
        """
        for attempt in range(3):
            response = self.session.get(
                self.resolved, headers={"Range": f"bytes={start}-{end}"}, timeout=120
            )
            if response.status_code in (403, 400) and attempt < 2:
                self.resolved = self._resolve()  # URL signée expirée
                continue
            if response.status_code != 206:
                raise OSError(f"Requête Range refusée (code {response.status_code})")
            self.requests_count += 1
            self.bytes_read += len(response.content)
            return response.content
        raise OSError("Échec de la requête Range")

    def readable(self) -> bool:
        """Le fichier est lisible."""
        return True

    def seekable(self) -> bool:
        """Le fichier est « seekable » (requis par zipfile)."""
        return True

    def tell(self) -> int:
        """Position courante."""
        return self.position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        """Déplace la position (début, courante ou fin).

        Args:
            offset: Décalage en octets.
            whence: ``SEEK_SET``, ``SEEK_CUR`` ou ``SEEK_END``.

        Returns:
            La nouvelle position.
        """
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self.position, io.SEEK_END: self.size}[whence]
        self.position = max(0, base + offset)
        return self.position

    def readinto(self, buffer) -> int:
        """Remplit ``buffer`` depuis la position courante (lecture par blocs avec cache).

        Args:
            buffer: Tampon à remplir.

        Returns:
            Nombre d'octets lus (0 en fin de fichier).
        """
        want = min(len(buffer), self.size - self.position)
        if want <= 0:
            return 0
        start, end = self.position, self.position + want
        if not (self.cache_start <= start and end <= self.cache_start + len(self.cache)):
            if want >= BLOCK:  # grande lecture : directe, sans cache
                buffer[:want] = self._fetch(start, end - 1)
                self.position = end
                return want
            self.cache = self._fetch(start, min(self.size, start + BLOCK) - 1)
            self.cache_start = start
            want = min(want, len(self.cache))
            end = start + want
        offset = start - self.cache_start
        buffer[:want] = self.cache[offset : offset + want]
        self.position = end
        return want


def group_patients(infos: list[zipfile.ZipInfo]) -> dict[str, dict[str, zipfile.ZipInfo]]:
    """Regroupe les fichiers utiles du zip par identifiant de patient.

    Args:
        infos: Entrées du répertoire central (les copies ``__MACOSX`` sont ignorées).

    Returns:
        ``{patient: {rôle: entrée}}`` avec les rôles ``t1``, ``t2`` et ``tumor``.
    """
    patients: dict[str, dict[str, zipfile.ZipInfo]] = {}
    for info in infos:
        match = NAME_PATTERN.fullmatch(info.filename)
        if match and not info.filename.startswith("__MACOSX") and match[2] in ROLES:
            patients.setdefault(match[1], {})[ROLES[match[2]]] = info
    return patients


def choose_patients(patients: dict[str, dict], count: int, seed: int = SEED) -> list[str]:
    """Tire au sort ``count`` patients qui ont T1, T2 ET masque (graine fixe).

    Args:
        patients: Sortie de :func:`group_patients`.
        count: Nombre de patients voulu.
        seed: Graine du tirage.

    Returns:
        Identifiants triés (ordre numérique).
    """
    eligible = sorted((p for p, f in patients.items() if {"t1", "t2", "tumor"} <= set(f)), key=int)
    return sorted(random.Random(seed).sample(eligible, min(count, len(eligible))), key=int)


def fetch_patient(
    archive: zipfile.ZipFile, files: dict[str, zipfile.ZipInfo], pid: str, dest: Path
) -> dict[str, Path]:
    """Télécharge les fichiers T1, T2 et masque d'un patient dans ``dest``.

    Args:
        archive: Zip distant ouvert sur un :class:`HttpRangeFile`.
        files: Entrées du patient (``{rôle: entrée}``).
        pid: Identifiant du patient.
        dest: Dossier de destination (créé si besoin).

    Returns:
        ``{rôle: chemin local}``.
    """
    dest.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for role, info in files.items():
        target = dest / f"{pid}_{role}.nii.gz"
        if not (target.is_file() and target.stat().st_size == info.file_size):
            with archive.open(info) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst, BLOCK)
        paths[role] = target
    return paths


def free_gb(path: Path) -> float:
    """Place libre (Go) sur le disque qui contient ``path``."""
    return shutil.disk_usage(path).free / 1e9


def main() -> None:
    """Point d'entrée : liste le zip ou télécharge l'échantillon."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--list", action="store_true", help="lire le répertoire central seulement")
    parser.add_argument("--download", action="store_true", help="télécharger l'échantillon")
    parser.add_argument("--patients", type=int, default=30, help="nombre de patients (30 max)")
    parser.add_argument("--min-free-gb", type=float, default=3.0, help="place libre minimale (Go)")
    parser.add_argument("--max-total-gb", type=float, default=1.0, help="volume total maximal (Go)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    remote = HttpRangeFile(FILE_URL, ZIP_SIZE)
    with zipfile.ZipFile(remote) as archive:
        infos = archive.infolist()
        LOGGER.info(
            "Répertoire central : %d entrées (%d requêtes, %.1f Mo lus)",
            len(infos),
            remote.requests_count,
            remote.bytes_read / 1e6,
        )
        patients = group_patients(infos)
        counts = {r: sum(r in f for f in patients.values()) for r in ("t1", "t2", "tumor")}
        LOGGER.info("%d patients ; fichiers par rôle : %s", len(patients), counts)
        for role in ("t1", "t2", "tumor"):
            sizes = sorted(f[role].file_size / 1e6 for f in patients.values() if role in f)
            if sizes:
                LOGGER.info(
                    "Taille %s (Mo) min/médiane/max : %.1f / %.1f / %.1f",
                    role,
                    sizes[0],
                    sizes[len(sizes) // 2],
                    sizes[-1],
                )
        if not args.download:
            return
        if args.patients > 30:
            raise SystemExit("30 patients au maximum (règle de l'étape).")
        total = 0.0
        for pid in choose_patients(patients, args.patients):
            for role, info in sorted(patients[pid].items()):
                target = OUTPUT_DIR / "volumes" / f"{pid}_{role}.nii.gz"
                if target.is_file() and target.stat().st_size == info.file_size:
                    continue  # déjà téléchargé
                size_gb = info.file_size / 1e9
                if free_gb(PROJECT_ROOT) - size_gb < args.min_free_gb:
                    raise SystemExit(f"Place libre insuffisante (< {args.min_free_gb} Go) : arrêt.")
                if total + size_gb > args.max_total_gb:
                    raise SystemExit(f"Plafond de {args.max_total_gb} Go atteint : arrêt.")
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(info) as src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst, BLOCK)
                total += size_gb
                LOGGER.info("%s (%.0f Mo), total %.2f Go", target.name, info.file_size / 1e6, total)
        LOGGER.info("Terminé : %.2f Go sur disque (%d requêtes).", total, remote.requests_count)


if __name__ == "__main__":
    main()
