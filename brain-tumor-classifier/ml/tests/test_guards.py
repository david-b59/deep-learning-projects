"""Tests de brain_mri.guards sur de petites données synthétiques (aucun vrai modèle chargé)."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from sklearn.covariance import LedoitWolf

from brain_mri import guards


def test_color_gray_is_zero() -> None:
    """Une image à un canal, ou RGB avec trois canaux identiques, vaut 0."""
    gray = np.full((20, 20), 120, dtype=np.uint8)
    assert guards.color_statistic(gray) == 0.0
    assert guards.color_statistic(np.stack([gray] * 3, axis=-1)) == 0.0


def test_color_ignores_dark_pixels() -> None:
    """Les pixels dont le canal max est <= 16 ne comptent pas (fond noir bruité)."""
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    image[..., 0] = 16  # rouge 16 : écart 16 mais canal max pas > 16
    assert guards.color_statistic(image) == 0.0
    assert guards.color_statistic(np.zeros((4, 4, 3), dtype=np.uint8)) == 0.0


def test_color_counts_colored_part() -> None:
    """Une moitié d'image rouge et une moitié grise donne 0,5 ; l'écart 24 exact ne compte pas."""
    image = np.full((10, 10, 3), 100, dtype=np.uint8)
    image[:5, :, 0] = 200  # écart 100 > 24 sur la moitié haute
    assert guards.color_statistic(image) == pytest.approx(0.5)
    edge = np.full((4, 4, 3), 100, dtype=np.uint8)
    edge[..., 0] = 124  # écart exactement 24 : pas strictement supérieur
    assert guards.color_statistic(edge) == 0.0


def test_color_threshold_floor_and_factor() -> None:
    """Seuil = 2 x maximum, mais jamais sous 1 %."""
    assert guards.color_threshold(0.0) == 0.01
    assert guards.color_threshold(0.002) == 0.01
    assert guards.color_threshold(0.03) == pytest.approx(0.06)


def make_blobs(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Fabrique 4 nuages gaussiens en dimension 12 (caractéristiques et classes factices)."""
    rng = np.random.default_rng(seed)
    labels = np.repeat(np.arange(4), 60)
    centers = rng.normal(0, 3, (4, 12))
    return centers[labels] + rng.normal(0, 1, (240, 12)), labels


def test_mahalanobis_matches_sklearn() -> None:
    """La distance du module égale celle de sklearn (LedoitWolf.mahalanobis) en espace ACP."""
    features, labels = make_blobs()
    fit = guards.fit_mahalanobis(features, labels, n_components=6)
    head = guards.GuardHead(12, 6, 4)
    for name, value in (
        ("pca_mean", fit.mean),
        ("pca_components", fit.components),
        ("precision", fit.precision),
        ("centers", fit.centers),
    ):
        getattr(head, name).copy_(torch.from_numpy(value))
    logits = torch.zeros(len(features), 4)
    _, distance, _ = head(logits, torch.from_numpy(features.astype(np.float32)))
    # Référence : mêmes résidus, LedoitWolf de sklearn, distance au centre le plus proche.
    projected = (features - fit.mean) @ fit.components.T
    residuals = projected - fit.centers[labels]
    lw = LedoitWolf(assume_centered=True).fit(residuals)
    squared = np.stack([lw.mahalanobis(projected - c) for c in fit.centers])  # carrés
    expected = np.sqrt(squared.min(axis=0))
    np.testing.assert_allclose(distance.numpy(), expected, rtol=1e-3, atol=1e-3)


def test_temperature_keeps_argmax_and_helps_overconfident() -> None:
    """La température ne change pas l'argmax ; elle dépasse 1 si le modèle est trop confiant."""
    rng = np.random.default_rng(1)
    labels = rng.integers(0, 4, 400)
    logits = rng.normal(0, 1, (400, 4)) + 8.0 * np.eye(4)[labels] * (rng.random((400, 1)) < 0.7)
    logits *= 3.0  # logits gonflés : modèle sur-confiant
    temperature = guards.fit_temperature(logits, labels)
    assert temperature > 1.0
    head = guards.GuardHead(4, 2, 4)
    head.temperature.fill_(temperature)
    probabilities, _, _ = head(torch.from_numpy(logits.astype(np.float32)), torch.zeros(400, 4))
    assert np.array_equal(probabilities.numpy().argmax(1), logits.argmax(1))
    before = torch.softmax(torch.from_numpy(logits), 1).numpy()
    assert guards.expected_calibration_error(probabilities.numpy(), labels) <= (
        guards.expected_calibration_error(before, labels)
    )


def test_verdict_order() -> None:
    """Ordre : couleur > T2 > hors distribution > confiance basse > accepté."""
    t = guards.GuardThresholds(color=0.05, tau_d=10.0, tau_p=0.6, tau_t2=0.5)
    assert guards.verdict(0.5, 0.9, 99.0, 0.1, t) == "color"
    assert guards.verdict(0.0, 0.9, 99.0, 0.1, t) == "t2_like"
    assert guards.verdict(0.0, 0.1, 99.0, 0.1, t) == "out_of_distribution"
    assert guards.verdict(0.0, 0.1, 1.0, 0.1, t) == "low_confidence"
    assert guards.verdict(0.0, 0.1, 1.0, 0.9, t) == "accepted"


def test_tau_choices_follow_rules() -> None:
    """tau_D = percentile 97 ; tau_P garde la part refusée <= 5 % ; tau_T2 respecte les 2 taux."""
    distances = np.arange(100, dtype=float)
    max_prob = np.linspace(0.3, 1.0, 100)
    tau_d = guards.choose_tau_d(distances)
    tau_p = guards.choose_tau_p(distances, max_prob, tau_d)
    assert guards.rejected_fraction(distances, max_prob, tau_d, tau_p) <= 0.05
    figshare = np.linspace(0, 0.1, 100)
    pandit = np.linspace(0, 0.3, 100)
    tau_t2 = guards.choose_tau_t2(figshare, pandit)
    assert (figshare > tau_t2).mean() <= 0.02 and (pandit > tau_t2).mean() <= 0.05


def test_fold_standardization_equivalent() -> None:
    """Replier la standardisation dans les poids donne le même score."""
    rng = np.random.default_rng(2)
    x = rng.normal(2, 3, (5, 7))
    w, b = rng.normal(size=7), 0.4
    mean, scale = x.mean(0), x.std(0)
    raw_w, raw_b = guards.fold_standardization(w, b, mean, scale)
    np.testing.assert_allclose(x @ raw_w + raw_b, ((x - mean) / scale) @ w + b)


def test_forbidden_data_assertion() -> None:
    """L'assertion refuse la partie test, les patients du test et le hors sujet."""
    ok = [{"part": "train"}, {"part": "val"}]
    guards.assert_no_forbidden_data(ok, {"3", "5"}, {"1", "16"}, {"train", "val"}, set())
    with pytest.raises(AssertionError):
        guards.assert_no_forbidden_data(ok + [{"part": "test"}], {"3"}, {"1"}, {"train"}, set())
    with pytest.raises(AssertionError):
        guards.assert_no_forbidden_data(ok, {"3", "16"}, {"1", "16"}, {"train"}, set())
    with pytest.raises(AssertionError):
        guards.assert_no_forbidden_data(ok, {"3"}, {"1"}, {"train", "test"}, set())
    with pytest.raises(AssertionError):
        guards.assert_no_forbidden_data(ok, {"3"}, {"1"}, {"train"}, {"abc"})


def test_save_and_load_roundtrip(tmp_path) -> None:
    """Sauvegarde safetensors puis relecture : mêmes sorties et mêmes seuils."""
    head = guards.GuardHead(12, 4, 4)
    head.temperature.fill_(1.7)
    head.pca_components.copy_(torch.randn(4, 12))
    head.thresholds.copy_(torch.tensor([0.02, 5.0, 0.6, 0.4]))
    guards.save_guards(head, tmp_path / "g.safetensors")
    loaded = guards.load_guards(tmp_path / "g.safetensors")
    logits, feats = torch.randn(3, 4), torch.randn(3, 12)
    for a, b in zip(head(logits, feats), loaded(logits, feats), strict=True):
        assert torch.allclose(a, b)
    assert loaded.threshold_values().tau_d == pytest.approx(5.0)
