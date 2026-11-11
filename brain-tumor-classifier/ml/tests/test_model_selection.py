"""Tests de brain_mri.model_selection : critères C1 à C5 et règle de décision de l'étape 11b."""

from __future__ import annotations

import numpy as np

from brain_mri.model_selection import (
    bootstrap_gap,
    bootstrap_mean,
    criteria,
    decide,
    model_metrics,
)

CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
STRICT = ("miroir", "bandes_noires_25", "zoom_0.85", "zoom_1.15", "recadrage_decentre_90")
LOOSE = ("etirement_+20", "etirement_-20", "jpeg_50")


def fake_invariance(pmram: float = 0.02, rates: dict[str, float] | None = None) -> dict:
    """Fabrique un JSON d'invariance minimal (même structure que evaluate_invariance.py).

    Args:
        pmram: Taux de bascule PMRAM.
        rates: Taux de bascule par transformation (0,01 par défaut).

    Returns:
        Le dictionnaire attendu par :func:`criteria`.
    """
    rates = {name: 0.01 for name in STRICT + LOOSE} | (rates or {})
    return {
        "transformations": {
            "transformations": {n: {"overall": {"flip_rate": r}} for n, r in rates.items()}
        },
        "pmram": {"all": {"safe": {"flip_rate": pmram, "n": 291}}},
    }


def fake_metrics(tumor_notumor: int = 10, recall: float = 0.99, gap: float = 0.0) -> dict:
    """Fabrique la partie de :func:`model_metrics` utilisée par les critères.

    Args:
        tumor_notumor: Tumeurs prédites « notumor ».
        recall: Rappel « notumor ».
        gap: Écart 512 - autres sur les tumeurs.

    Returns:
        Un dictionnaire de mesures minimal.
    """
    return {
        "tumor_predicted_notumor": tumor_notumor,
        "notumor_recall": recall,
        "tumor_gap_512_minus_other": {"value": gap},
    }


def test_bootstrap_mean_is_reproducible_and_brackets_the_value() -> None:
    """Même graine, même intervalle ; la valeur est dans l'intervalle."""
    values = np.array([1, 0, 1, 1, 0, 1, 1, 1], dtype=bool)
    first, second = bootstrap_mean(values), bootstrap_mean(values)
    assert first == second
    assert first["ci_low"] <= first["value"] <= first["ci_high"]
    assert bootstrap_mean(np.array([], dtype=bool))["value"] is None


def test_bootstrap_gap_sign() -> None:
    """Groupe A toujours juste, groupe B toujours faux : écart de +1 sans incertitude."""
    correct = np.array([True, True, False, False])
    group = np.array([True, True, False, False])
    gap = bootstrap_gap(correct, group)
    assert gap["value"] == 1.0 and gap["ci_low"] == 1.0
    empty = bootstrap_gap(correct, np.ones(4, dtype=bool))  # groupe B vide
    assert empty["value"] is None and empty["n_b"] == 0


def test_model_metrics_counts_tumors_predicted_notumor() -> None:
    """Une tumeur prédite « notumor » est comptée ; le rappel notumor est calculé."""
    labels = np.array([0, 1, 2, 2, 3])
    predictions = np.array([2, 1, 2, 0, 3])  # 1 tumeur -> notumor, 1 notumor raté
    is_512 = np.array([True, False, True, False, True])
    metrics = model_metrics(labels, predictions, CLASSES, is_512, np.ones(5, dtype=bool))
    assert metrics["tumor_predicted_notumor"] == 1
    assert metrics["notumor_recall"] == 0.5
    assert metrics["accuracy"]["value"] == 0.6


def test_criteria_pass_and_fail() -> None:
    """Un modèle sage passe tout ; chaque seuil dépassé fait échouer son critère."""
    ref = fake_metrics(tumor_notumor=15)
    diff = {"value": 0.04, "ci_low": 0.01, "ci_high": 0.07}
    good = criteria(fake_metrics(), fake_invariance(), ref, diff)
    assert all(good[c]["passed"] for c in ("C1", "C2", "C3", "C4", "C5"))

    bad = criteria(
        fake_metrics(tumor_notumor=22, gap=-0.12),
        fake_invariance(pmram=0.13, rates={"miroir": 0.054, "jpeg_50": 0.09}),
        ref,
        {"value": 0.04, "ci_low": -0.01, "ci_high": 0.09},
    )
    assert not bad["C1"]["passed"]  # IC de l'écart contient 0
    assert not bad["C2"]["passed"]
    assert bad["C3"]["failed"] == ["miroir"]  # JPEG à 9 % reste sous 10 %
    assert not bad["C4"]["passed"]
    assert not bad["C5"]["passed"]
    assert criteria(fake_metrics(), fake_invariance(), ref, None)["C1"]["passed"] is None


def test_decide_follows_the_rule() -> None:
    """Aucun éligible : modèle en ligne ; à moins de 2 points : le moins de bascules PMRAM."""
    ok = {c: {"passed": True} for c in ("C2", "C3", "C4")}
    ko = ok | {"C3": {"passed": False}}
    none = decide({"a": {"accuracy": 0.9, "pmram_flip_rate": 0.1, "criteria": ko}})
    assert none["winner"] is None
    models = {
        "x": {"accuracy": 0.91, "pmram_flip_rate": 0.08, "criteria": ok},
        "y": {"accuracy": 0.90, "pmram_flip_rate": 0.02, "criteria": ok},
        "z": {"accuracy": 0.95, "pmram_flip_rate": 0.01, "criteria": ko},
    }
    assert decide(models)["winner"] == "y"  # x et y à moins de 2 points, y bascule moins
    models["y"]["accuracy"] = 0.85
    assert decide(models)["winner"] == "x"
