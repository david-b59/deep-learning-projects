"""Tests de ``brain_mri.grouping`` sur de petites données synthétiques (étape 11b-1)."""

from __future__ import annotations

import numpy as np

from brain_mri.grouping import (
    build_groups,
    group_split,
    label_conflicts,
    match_patient_ids,
    residual_leak,
)


def unit(vectors: list[list[float]]) -> np.ndarray:
    """Normalise des vecteurs en norme 1 (comme les embeddings réels)."""
    array = np.array(vectors, dtype=np.float32)
    return array / np.linalg.norm(array, axis=1, keepdims=True)


def test_match_patient_ids_same_class_only() -> None:
    """Une copie figshare de même classe donne son patient ; une autre classe ne compte pas."""
    kaggle = unit([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    kaggle_labels = np.array([0, 1, 2])
    figshare = unit([[1, 0.01, 0], [0, 1, 0.01]])
    figshare_labels = np.array([0, 0])  # la 2e image figshare est de classe 0, pas 1
    patients, best = match_patient_ids(
        kaggle, kaggle_labels, figshare, figshare_labels, ["p1", "p2"], threshold=0.98
    )
    assert patients == ["p1", "", ""]  # l'image 1 ressemble à p2 mais d'une autre classe
    assert best[2] == -np.inf  # la classe 2 n'existe pas dans figshare


def test_label_conflicts_finds_only_cross_class_pairs() -> None:
    """Deux images quasi identiques de classes différentes sont signalées."""
    emb = unit([[1, 0], [1, 0.001], [0, 1], [0.001, 1]])
    labels = np.array([0, 1, 2, 2])  # 0 et 1 : conflit ; 2 et 3 : même classe
    pairs = label_conflicts(emb, labels, threshold=0.98)
    assert [(i, j) for i, j, _ in pairs] == [(0, 1)]


def test_build_groups_links_by_cosine_patient_and_sha() -> None:
    """Les liens cosinus (même classe), patient et SHA-256 se propagent de proche en proche."""
    emb = unit([[1, 0, 0], [1, 0.01, 0], [0, 1, 0], [0, 0, 1], [0.5, 0.5, 0.7], [1, 0.02, 0]])
    labels = np.array(
        [0, 0, 1, 1, 2, 1]
    )  # l'image 5 ressemble à 0 et 1 mais est d'une autre classe
    patients = ["", "", "", "", "", ""]
    patients[2] = patients[3] = "pX"  # même patient : 2 et 3 réunies
    sha = ["a", "b", "c", "d", "e", "e"]  # 4 et 5 identiques à l'octet près
    groups = build_groups(emb, labels, 0.98, patients, sha)
    assert groups[0] == groups[1]  # lien par cosinus
    assert groups[2] == groups[3]  # lien par patient
    assert groups[4] == groups[5]  # lien par SHA-256
    assert len({groups[0], groups[2], groups[4]}) == 3  # trois groupes distincts


def test_group_split_keeps_groups_whole_and_covers_all_parts() -> None:
    """Aucun groupe n'est coupé ; les trois parties existent ; le résultat est reproductible."""
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(70), 4)  # 70 groupes de 4 images
    labels = (groups % 2).astype(np.int64)  # un groupe = une seule classe
    extra = rng.integers(0, 2, size=70)[groups]
    parts = group_split(labels, groups, extra, n_splits=7, seed=42)
    for group in range(70):
        assert len(set(parts[groups == group])) == 1
    assert set(parts) == {"train", "val", "test"}
    assert (parts == "train").sum() > (parts == "test").sum() * 3
    assert np.array_equal(parts, group_split(labels, groups, extra, n_splits=7, seed=42))


def test_residual_leak_counts_by_bin() -> None:
    """Les images de test proches du train sont comptées, par tranche de cosinus."""
    train = unit([[1, 0], [0, 1]])
    test = unit([[1, 0.001], [1, 1]])  # la 1re copie le train ; la 2e est à 45 degrés (cos 0,71)
    result = residual_leak(test, train, threshold=0.98)
    assert result["n_test"] == 2
    assert result["n_above_threshold"] == 1
    assert result["by_cosine_bin"][">=0,98"] == 1
    assert result["by_cosine_bin"]["<0,90"] == 1
