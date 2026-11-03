"""Detect duplicate images, inside and between the Training and Testing folders.

Why it matters: if the same image is in both Training and Testing, the model is
"tested" on something it has already seen, and the test score is inflated.
Two files are duplicates here when their bytes are identical (same SHA-256).
Near-duplicates (same scan re-saved or resized) are NOT detected: this is a
documented limit of this check.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

LOGGER = logging.getLogger(__name__)

IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png"})
TRAIN_DIR_NAME = "Training"
TEST_DIR_NAME = "Testing"
REPORT_NAME = "duplicates_report.json"


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """Return the SHA-256 hex digest of a file, read by chunks (low memory)."""
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        while chunk := file.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def hash_images(root: Path) -> dict[str, list[str]]:
    """Hash every image under ``root``.

    Args:
        root: Folder to scan recursively.

    Returns:
        A mapping ``digest -> [relative POSIX paths]``, sorted for determinism.

    Raises:
        FileNotFoundError: If ``root`` does not exist.
    """
    if not root.is_dir():
        raise FileNotFoundError(f"Folder not found: {root}")
    groups: dict[str, list[str]] = defaultdict(list)
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            groups[sha256_file(path)].append(path.relative_to(root).as_posix())
    return dict(groups)


@dataclass
class DuplicateReport:
    """Result of the duplicate check. All paths are relative to the raw folder."""

    n_train: int = 0
    n_test: int = 0
    # Test files whose content also exists in Training (data leakage).
    test_files_in_train: list[str] = field(default_factory=list)
    # Groups of identical files inside Training (can leak into validation).
    train_groups: list[list[str]] = field(default_factory=list)
    # Groups of identical files inside Testing.
    test_groups: list[list[str]] = field(default_factory=list)

    def files_to_exclude(self) -> set[str]:
        """Files the data pipeline must not use.

        - Test files that also exist in Training (leakage into the test score).
        - Extra copies inside a group of identical files: the first path (sorted)
          is kept, the others are dropped. This keeps copies of one image from
          landing on both sides of the train/validation split.
        """
        excluded = set(self.test_files_in_train)
        for group in (*self.train_groups, *self.test_groups):
            excluded.update(sorted(group)[1:])
        return excluded

    def to_json(self) -> str:
        """Serialize the report to a readable JSON string."""
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, text: str) -> DuplicateReport:
        """Rebuild a report from the output of :meth:`to_json`."""
        return cls(**json.loads(text))


def find_duplicates(raw_dir: Path) -> DuplicateReport:
    """Compare the images of ``Training`` and ``Testing`` by content.

    Args:
        raw_dir: Folder that contains ``Training/`` and ``Testing/``.

    Returns:
        The duplicate report.
    """
    train = hash_images(raw_dir / TRAIN_DIR_NAME)
    test = hash_images(raw_dir / TEST_DIR_NAME)

    # Paths are stored relative to raw_dir, so prefix them with their split folder.
    def prefixed(split: str, paths: list[str]) -> list[str]:
        return [f"{split}/{p}" for p in paths]

    leaked = sorted(
        path
        for digest, paths in test.items()
        if digest in train
        for path in prefixed(TEST_DIR_NAME, paths)
    )
    report = DuplicateReport(
        n_train=sum(len(p) for p in train.values()),
        n_test=sum(len(p) for p in test.values()),
        test_files_in_train=leaked,
        train_groups=[prefixed(TRAIN_DIR_NAME, p) for p in train.values() if len(p) > 1],
        test_groups=[prefixed(TEST_DIR_NAME, p) for p in test.values() if len(p) > 1],
    )
    LOGGER.info(
        "Duplicates: %d test file(s) also in Training, %d group(s) inside Training, "
        "%d group(s) inside Testing",
        len(report.test_files_in_train),
        len(report.train_groups),
        len(report.test_groups),
    )
    return report


def save_report(report: DuplicateReport, processed_dir: Path) -> Path:
    """Write the report as JSON in ``processed_dir`` and return its path."""
    processed_dir.mkdir(parents=True, exist_ok=True)
    path = processed_dir / REPORT_NAME
    path.write_text(report.to_json(), encoding="utf-8")
    return path


def count_images(root: Path) -> int:
    """Count the image files under ``root`` (no hashing, so it is fast)."""
    return sum(1 for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def load_or_create_report(raw_dir: Path, processed_dir: Path) -> DuplicateReport:
    """Load the cached report, or compute and cache it.

    The cache is recomputed when it is missing or when its image counts no longer
    match the files on disk (data changed or downloaded again).
    """
    path = processed_dir / REPORT_NAME
    if path.is_file():
        cached = DuplicateReport.from_json(path.read_text(encoding="utf-8"))
        if cached.n_train == count_images(
            raw_dir / TRAIN_DIR_NAME
        ) and cached.n_test == count_images(raw_dir / TEST_DIR_NAME):
            return cached
        LOGGER.warning("Cached duplicate report is out of date: recomputing")
    report = find_duplicates(raw_dir)
    save_report(report, processed_dir)
    return report
