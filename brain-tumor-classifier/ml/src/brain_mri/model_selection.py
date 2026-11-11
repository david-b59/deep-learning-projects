"""Comparaison des modèles candidats et règle de décision de l'étape 11b (11b-5).

Les critères ont été écrits AVANT toute mesure (plan révisé du 2026-11-09) et sont appliqués
ici mécaniquement, sans jugement au cas par cas :

    - C1 : exactitude >= celle de A0 + 3 points, et l'intervalle de confiance de l'écart
      (bootstrap apparié) ne contient pas 0 ;
    - C2 : taux de bascule PMRAM (sous-ensemble ``seuil_2``, copies sûres) <= 10 % ;
    - C3 : taux de bascule <= 5 % pour miroir, bandes noires, zooms et recadrage décentré ;
      <= 10 % pour les étirements et le JPEG 50 ;
    - C4 : tumeurs prédites « notumor » <= celles de A0 ET rappel « notumor » >= 95 % ;
    - C5 : écart d'exactitude entre tumeurs 512 x 512 et tumeurs d'autres tailles < 10 points ;
    - C6 (parité navigateur, taille ONNX) : mesuré seulement pour le gagnant, hors de ce module.

Règle de choix : parmi les modèles qui passent C2, C3 et C4, le plus exact ; si deux sont à
moins de 2 points, celui qui bascule le moins sur PMRAM ; si aucun ne passe, on garde le
modèle en ligne et on publie les chiffres honnêtes.
"""

from __future__ import annotations

import numpy as np

# Seuils des critères (fixés avant mesure)
C1_MIN_GAIN = 0.03  # +3 points d'exactitude par rapport à A0
C2_MAX_PMRAM_FLIPS = 0.10  # bascules PMRAM maximales
C3_STRICT = ("miroir", "bandes_noires_25", "zoom_0.85", "zoom_1.15", "recadrage_decentre_90")
C3_LOOSE = ("etirement_+20", "etirement_-20", "jpeg_50")
C3_STRICT_MAX = 0.05  # bascules maximales pour les transformations « fortes »
C3_LOOSE_MAX = 0.10  # bascules maximales pour étirement et JPEG
C4_MIN_NOTUMOR_RECALL = 0.95  # rappel minimal de la classe « notumor »
C5_MAX_GAP = 0.10  # écart maximal 512 / autres tailles (tumeurs seulement)
TIE_MARGIN = 0.02  # deux exactitudes à moins de 2 points sont jugées à égalité
NOTUMOR = "notumor"


def bootstrap_mean(values: np.ndarray, draws: int = 2000, seed: int = 42) -> dict:
    """Moyenne et intervalle de confiance à 95 % par bootstrap (percentiles).

    Args:
        values: Valeurs par image (booléens « juste » ou différences -1 / 0 / +1).
        draws: Nombre de tirages avec remise.
        seed: Graine du générateur (résultat reproductible).

    Returns:
        ``{"n", "value", "ci_low", "ci_high"}`` ; valeurs ``None`` si ``values`` est vide.
    """
    n = len(values)
    if n == 0:
        return {"n": 0, "value": None, "ci_low": None, "ci_high": None}
    rng = np.random.default_rng(seed)
    means = values.astype(np.float64)[rng.integers(0, n, size=(draws, n))].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return {"n": n, "value": float(values.mean()), "ci_low": float(low), "ci_high": float(high)}


def bootstrap_gap(
    correct: np.ndarray, group: np.ndarray, draws: int = 2000, seed: int = 42
) -> dict:
    """Écart d'exactitude entre deux groupes d'images, avec IC à 95 % par bootstrap.

    Chaque tirage rééchantillonne les deux groupes séparément (tailles conservées).

    Args:
        correct: Booléens « le modèle a juste », ``(N,)``.
        group: Booléens « l'image est dans le groupe A » (les autres forment le groupe B).
        draws: Nombre de tirages.
        seed: Graine du générateur.

    Returns:
        ``{"n_a", "n_b", "value", "ci_low", "ci_high"}`` où ``value`` = exactitude A - B.
    """
    a = correct[group].astype(np.float64)
    b = correct[~group].astype(np.float64)
    if len(a) == 0 or len(b) == 0:  # un groupe vide : aucun écart mesurable
        return {"n_a": len(a), "n_b": len(b), "value": None, "ci_low": None, "ci_high": None}
    rng = np.random.default_rng(seed)
    means_a = a[rng.integers(0, len(a), size=(draws, len(a)))].mean(axis=1)
    means_b = b[rng.integers(0, len(b), size=(draws, len(b)))].mean(axis=1)
    low, high = np.percentile(means_a - means_b, [2.5, 97.5])
    return {
        "n_a": len(a),
        "n_b": len(b),
        "value": float(a.mean() - b.mean()),
        "ci_low": float(low),
        "ci_high": float(high),
    }


def model_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    class_names: list[str],
    is_512: np.ndarray,
    strict: np.ndarray,
) -> dict:
    """Mesures d'un modèle sur le test : exactitudes avec IC, erreurs graves, écart de taille.

    Args:
        labels: Vraies classes (indices), ``(N,)``.
        predictions: Classes prédites (indices), ``(N,)``.
        class_names: Noms des classes, dans l'ordre des indices.
        is_512: Booléens « l'image fait 512 x 512 ».
        strict: Booléens « l'image appartient au sous-test strict » (aucune voisine >= 0,94).

    Returns:
        Exactitude (test complet et strict, globale et par classe), tumeurs prédites
        « notumor », rappel « notumor » et écart 512 / autres sur les tumeurs.
    """
    correct = predictions == labels
    notumor = class_names.index(NOTUMOR)
    tumor = labels != notumor
    return {
        "accuracy": bootstrap_mean(correct),
        "accuracy_strict": bootstrap_mean(correct[strict]),
        "per_class": {
            name: bootstrap_mean(correct[labels == i]) for i, name in enumerate(class_names)
        },
        "tumor_predicted_notumor": int((tumor & (predictions == notumor)).sum()),
        "notumor_recall": float(correct[labels == notumor].mean()),
        "tumor_gap_512_minus_other": bootstrap_gap(correct[tumor], is_512[tumor]),
    }


def criteria(
    metrics: dict,
    invariance: dict,
    reference_metrics: dict,
    diff_vs_reference: dict | None,
) -> dict:
    """Applique les critères C1 à C5 à un modèle.

    Args:
        metrics: Sortie de :func:`model_metrics` pour ce modèle.
        invariance: JSON d'invariance du modèle (``transformations`` et ``pmram``).
        reference_metrics: Sortie de :func:`model_metrics` pour A0 (la référence).
        diff_vs_reference: Écart apparié avec A0 (sortie de :func:`bootstrap_mean` sur
            ``juste_modèle - juste_A0``), ou ``None`` pour A0 lui-même.

    Returns:
        ``{"C1": {...}, ..., "C5": {...}}``, chacun avec les valeurs utiles et ``passed``
        (``None`` quand le critère ne s'applique pas, ex. C1 pour A0).
    """
    if diff_vs_reference is None:
        c1 = {"passed": None, "reason": "modèle de référence"}
    else:
        c1 = {
            "gain": diff_vs_reference["value"],
            "ci": [diff_vs_reference["ci_low"], diff_vs_reference["ci_high"]],
            "passed": diff_vs_reference["value"] >= C1_MIN_GAIN and diff_vs_reference["ci_low"] > 0,
        }
    pmram = invariance["pmram"]["all"]["safe"]
    rates = {
        name: block["overall"]["flip_rate"]
        for name, block in invariance["transformations"]["transformations"].items()
    }
    failed = [n for n in C3_STRICT if rates[n] > C3_STRICT_MAX] + [
        n for n in C3_LOOSE if rates[n] > C3_LOOSE_MAX
    ]
    gap = metrics["tumor_gap_512_minus_other"]
    return {
        "C1": c1,
        "C2": {
            "pmram_flip_rate": pmram["flip_rate"],
            "n": pmram["n"],
            "passed": pmram["flip_rate"] <= C2_MAX_PMRAM_FLIPS,
        },
        "C3": {"flip_rates": rates, "failed": failed, "passed": not failed},
        "C4": {
            "tumor_predicted_notumor": metrics["tumor_predicted_notumor"],
            "reference": reference_metrics["tumor_predicted_notumor"],
            "notumor_recall": metrics["notumor_recall"],
            "passed": metrics["tumor_predicted_notumor"]
            <= reference_metrics["tumor_predicted_notumor"]
            and metrics["notumor_recall"] >= C4_MIN_NOTUMOR_RECALL,
        },
        "C5": {
            "gap": gap,
            "passed": None if gap["value"] is None else abs(gap["value"]) < C5_MAX_GAP,
        },
    }


def decide(models: dict[str, dict]) -> dict:
    """Applique la règle de choix aux modèles évalués.

    Args:
        models: ``{nom: {"accuracy": float, "pmram_flip_rate": float, "criteria": {...}}}``.

    Returns:
        ``{"eligible": [...], "winner": nom ou None, "decision": texte en français}``.
    """
    eligible = [
        name
        for name, m in models.items()
        if all(m["criteria"][c]["passed"] for c in ("C2", "C3", "C4"))
    ]
    if not eligible:
        return {
            "eligible": [],
            "winner": None,
            "decision": "aucun modèle ne passe C2, C3 et C4 : on garde le modèle en ligne "
            "et on publie les chiffres honnêtes",
        }
    best = max(models[n]["accuracy"] for n in eligible)
    close = [n for n in eligible if best - models[n]["accuracy"] < TIE_MARGIN]
    winner = min(close, key=lambda n: models[n]["pmram_flip_rate"])
    why = (
        "le plus exact parmi les modèles qui passent C2, C3 et C4"
        if len(close) == 1
        else "exactitudes à moins de 2 points : celui qui bascule le moins sur PMRAM"
    )
    return {"eligible": eligible, "winner": winner, "decision": why}
