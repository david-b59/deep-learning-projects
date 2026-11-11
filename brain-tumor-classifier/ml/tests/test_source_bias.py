"""Tests de brain_mri.source_bias : tableau classe x source, poids de groupe, critères S1 à S3."""

from __future__ import annotations

import numpy as np

from brain_mri.source_bias import group_table, group_weights, source_criteria

CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]


def small_case() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Petit test fait main : 4 gliomes (2 en 512), 2 notumor hors 512.

    Returns:
        ``(labels, predictions, is_512)``. Les gliomes hors 512 : un juste, un prédit notumor.
    """
    labels = np.array([0, 0, 0, 0, 2, 2])
    predictions = np.array([0, 0, 0, 2, 2, 2])
    is_512 = np.array([True, True, False, False, False, False])
    return labels, predictions, is_512


def test_group_table_counts_and_rates() -> None:
    """Le tableau donne les effectifs, l'exactitude et les tumeurs prédites notumor."""
    labels, predictions, is_512 = small_case()
    rows = {(r["class"], r["is_512"]): r for r in group_table(labels, predictions, is_512, CLASSES)}
    assert set(rows) == {("glioma", True), ("glioma", False), ("notumor", False)}
    assert rows[("glioma", True)]["accuracy"]["value"] == 1.0
    assert rows[("glioma", False)]["n"] == 2
    assert rows[("glioma", False)]["accuracy"]["value"] == 0.5
    assert rows[("glioma", False)]["predicted_notumor"] == 1
    assert rows[("glioma", False)]["predicted_notumor_rate"] == 0.5
    assert rows[("notumor", False)]["predicted_notumor"] is None  # sans objet pour notumor


def test_group_weights_equal_total_and_floor() -> None:
    """Un groupe de 40 pèse 1 au total ; un groupe de 10 compte comme 30 images."""
    labels = np.array([0] * 40 + [2] * 10)
    is_512 = np.array([True] * 40 + [False] * 10)
    weights = group_weights(labels, is_512, min_count=30)
    assert np.isclose(weights[:40].sum(), 1.0)
    assert np.isclose(weights[40:].sum(), 10 / 30)  # plancher à 30 images


def test_source_criteria_pass() -> None:
    """Un modèle parfait passe S1, S2 et les garde-fous ; S3 suit l'estimation hors pli."""
    labels = np.array([0] * 10 + [2] * 10)
    is_512 = np.array([False] * 20)
    reference = labels.copy()
    reference[:5] = 2  # A0 rate 5 gliomes
    result = source_criteria(labels, labels.copy(), is_512, CLASSES, reference, np.ones(50, bool))
    assert result["S1"]["passed"] is True
    assert result["S2"]["passed"] is True
    assert result["S3"]["passed"] is True
    assert result["guards"]["passed"] is True


def test_source_criteria_fail_and_none() -> None:
    """Trop de tumeurs prédites notumor fait échouer S2 ; A0 et l'absence d'OOF donnent None."""
    labels = np.array([0] * 10 + [2] * 10)
    is_512 = np.array([False] * 20)
    predictions = labels.copy()
    predictions[:6] = 2  # 6 tumeurs -> notumor
    result = source_criteria(labels, predictions, is_512, CLASSES, None, None)
    assert result["S2"]["passed"] is False
    assert result["S1"]["passed"] is None  # modèle de référence
    assert result["S3"]["passed"] is None  # pas d'estimation hors pli
    assert result["guards"]["passed"] is False  # exactitude 70 % < 87 %
