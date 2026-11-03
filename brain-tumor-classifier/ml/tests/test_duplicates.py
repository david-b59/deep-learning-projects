"""Tests for brain_mri.duplicates."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from brain_mri.duplicates import (
    DuplicateReport,
    find_duplicates,
    hash_images,
    load_or_create_report,
    sha256_file,
)


def test_sha256_is_same_for_identical_files(raw_dir: Path) -> None:
    a = raw_dir / "Training/glioma/img_0.jpg"
    b = raw_dir / "Testing/glioma/leak.jpg"
    assert sha256_file(a) == sha256_file(b)
    assert sha256_file(a) != sha256_file(raw_dir / "Training/glioma/img_1.jpg")


def test_hash_images_missing_folder_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        hash_images(tmp_path / "does_not_exist")


def test_find_duplicates_detects_planted_copies(raw_dir: Path) -> None:
    report = find_duplicates(raw_dir)
    assert report.test_files_in_train == ["Testing/glioma/leak.jpg"]
    assert [sorted(g) for g in report.train_groups] == [
        ["Training/notumor/copy.jpg", "Training/notumor/img_0.jpg"]
    ]
    assert report.test_groups == []
    assert report.n_train == 81 and report.n_test == 25


def test_files_to_exclude_keeps_one_copy_per_group(raw_dir: Path) -> None:
    excluded = find_duplicates(raw_dir).files_to_exclude()
    # leaked test file removed; of the two identical training files, only one is dropped
    assert "Testing/glioma/leak.jpg" in excluded
    dropped = {"Training/notumor/copy.jpg", "Training/notumor/img_0.jpg"} & excluded
    assert len(dropped) == 1
    assert len(excluded) == 2


def test_stale_cache_is_recomputed(raw_dir: Path, tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    first = load_or_create_report(raw_dir, processed)
    # A new image appears on disk: the cached counts no longer match the files.
    shutil.copy(raw_dir / "Training/glioma/img_1.jpg", raw_dir / "Training/glioma/new_copy.jpg")
    second = load_or_create_report(raw_dir, processed)
    assert second.n_train == first.n_train + 1
    assert "Training/glioma/new_copy.jpg" in {p for g in second.train_groups for p in g}


def test_report_json_round_trip_and_cache(raw_dir: Path, tmp_path: Path) -> None:
    report = find_duplicates(raw_dir)
    assert DuplicateReport.from_json(report.to_json()) == report
    processed = tmp_path / "processed"
    first = load_or_create_report(raw_dir, processed)
    assert (processed / "duplicates_report.json").is_file()
    assert load_or_create_report(raw_dir, processed) == first
