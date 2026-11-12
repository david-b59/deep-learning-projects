"""Tests de la logique pure des scripts de données de l'étape 11c-0 (tirage, échantillonnage)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import download_offtopic as offtopic  # noqa: E402
import extract_pandit_dev as dev  # noqa: E402


def test_dev_draw_is_disjoint_from_test() -> None:
    """Les patients de développement et la réserve ne contiennent aucun patient du test."""
    eligible = [str(i) for i in range(1, 101)]
    test = [str(i) for i in range(1, 31)]
    chosen, reserve = dev.draw_dev_patients(eligible, test, count=40)
    assert len(chosen) == 40
    assert not set(chosen) & set(test)
    assert not set(reserve) & set(test)
    assert not set(chosen) & set(reserve)
    assert len(chosen) + len(reserve) == len(eligible) - len(test)


def test_dev_draw_is_reproducible_and_sorted() -> None:
    """Même graine, même tirage ; le résultat est trié par ordre numérique."""
    eligible = [str(i) for i in range(1, 60)]
    first, _ = dev.draw_dev_patients(eligible, ["3", "9"], count=10)
    second, _ = dev.draw_dev_patients(eligible, ["3", "9"], count=10)
    assert first == second
    assert first == sorted(first, key=int)


def test_dev_draw_caps_at_pool_size() -> None:
    """Si le vivier est plus petit que le nombre voulu, on prend tout le vivier."""
    chosen, reserve = dev.draw_dev_patients(["1", "2", "3"], ["1"], count=40)
    assert chosen == ["2", "3"]
    assert reserve == []


def test_sample_indices_is_deterministic_and_distinct() -> None:
    """L'échantillon est trié, sans doublon, reproductible, et plafonné à la population."""
    sample = offtopic.sample_indices(1000, 100)
    assert sample == offtopic.sample_indices(1000, 100)
    assert len(set(sample)) == 100 and sample == sorted(sample)
    assert offtopic.sample_indices(5, 100) == [0, 1, 2, 3, 4]


@pytest.mark.parametrize(
    ("licence", "expected"),
    [
        ("CC0", True),
        ("Public domain", True),
        ("CC BY 4.0", True),
        ("CC BY 2.0", True),
        ("CC BY-SA 4.0", False),
        ("CC BY-NC 2.0", False),
        ("CC BY-ND 2.0", False),
        ("GFDL", False),
        ("", False),
    ],
)
def test_license_filter(licence: str, expected: bool) -> None:
    """Seules CC0, domaine public et CC BY (sans SA, NC, ND) sont acceptées."""
    assert offtopic.is_allowed_license(licence) is expected


def test_medmnist_never_includes_dermamnist() -> None:
    """DermaMNIST (CC BY-NC) n'est dans aucune catégorie."""
    assert all("dermamnist" not in name for name, _ in offtopic.MEDMNIST.values())
