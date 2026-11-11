"""Rapport du modèle b_dfr : critères C1 à C5, S1 à S3 et garde-fous (étape 11b-6b).

Lit les mesures déjà produites (``docs/figures/b_dfr/evaluation.json``,
``invariance_head_crop.json`` et ``cv.json``) et applique les mêmes règles que
``compare_models.py`` (C1 à C5, écrits avant mesure) et ``audit_source_bias_models.py``
(S1 à S3, garde-fous). Les anciens rapports
(``model_comparison_11b.json``, ``source_bias_11b.json``) ne sont PAS modifiés : tout est écrit
dans ``docs/figures/dfr_11b.json``.

Utilisation (depuis ``brain-tumor-classifier/ml/``, après ``train_dfr.py``, ``evaluate.py`` et
``evaluate_invariance.py`` sur b_dfr) ::

    PYTHONUTF8=1 uv run python scripts/report_dfr.py
"""

from __future__ import annotations

import json

import audit_source_bias_models as audit
import compare_models as compare
import numpy as np

from brain_mri.model_selection import bootstrap_mean, criteria, model_metrics
from brain_mri.source_bias import (
    S3_MIN_ACCURACY,
    S3_MIN_CI_LOW,
    group_table,
    source_criteria,
)

NAME = "b_dfr"


def main() -> None:
    """Applique les critères à b_dfr, écrit ``dfr_11b.json`` et résume en français."""
    files, labels, is_512, strict, class_names = compare.load_test_set()
    reference = compare.REFERENCE
    predictions = {
        n: compare.load_predictions(n, files, labels, class_names) for n in (reference, NAME)
    }
    metrics = {
        n: model_metrics(labels, p, class_names, is_512, strict) for n, p in predictions.items()
    }
    diff = bootstrap_mean(
        (predictions[NAME] == labels).astype(np.int64) - (predictions[reference] == labels)
    )
    invariance_file = compare.EXTRA_MODELS[NAME][1]
    invariance = json.loads(
        (compare.FIGURES_DIR / NAME / invariance_file).read_text(encoding="utf-8")
    )
    checks = criteria(metrics[NAME], invariance, metrics[reference], diff)

    cv = json.loads((compare.FIGURES_DIR / NAME / "cv.json").read_text(encoding="utf-8"))
    s3_cv = cv["s3_tumors_not_512_oof"]["accuracy"]
    source = source_criteria(
        labels, predictions[NAME], is_512, class_names, predictions[reference], None
    )
    source["S3"] = {  # estimation hors pli de train_dfr.py (cv.json), mêmes seuils que source_bias
        "oof_accuracy": s3_cv,
        "passed": bool(s3_cv["value"] >= S3_MIN_ACCURACY and s3_cv["ci_low"] >= S3_MIN_CI_LOW),
    }
    # Recalcul des prédictions depuis la tête enregistrée : doit concorder avec evaluation.json.
    feature_set, checkpoint, prefix = audit.EXTRA_MODELS[NAME]
    recomputed = audit.recompute_predictions(
        checkpoint, prefix, audit.load_features(feature_set, "test")
    )
    report = {
        "model": NAME,
        "weight_decay": cv["selected_weight_decay"],
        "test": {"n": len(labels), "n_not_512": int((~is_512).sum())},
        "metrics": metrics[NAME],
        "diff_vs_a0": diff,
        "criteria_c1_c5": checks,
        "source_table": group_table(labels, predictions[NAME], is_512, class_names),
        "source_criteria": source,
        "concordance": {
            "same_prediction_rate": float((recomputed == predictions[NAME]).mean()),
        },
        "cv": {"selected_weight_decay": cv["selected_weight_decay"], "grid": cv["grid"]},
    }
    out = compare.FIGURES_DIR / "dfr_11b.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    acc = metrics[NAME]["accuracy"]
    print(
        f"{NAME} : exactitude {compare.pct(acc['value'])} "
        f"[{compare.pct(acc['ci_low'])} ; {compare.pct(acc['ci_high'])}]"
    )
    print("C1-C5 :", {c: v["passed"] for c, v in checks.items()})
    print("S1-S3 :", {c: source[c]["passed"] for c in ("S1", "S2", "S3")}, source["guards"])
    for row in report["source_table"]:
        print(
            f"  {row['class']:<11} {'512' if row['is_512'] else 'autre':<6} n={row['n']:<4} "
            f"exactitude {compare.pct(row['accuracy']['value'])} "
            f"-> notumor {row['predicted_notumor']}"
        )
    print(f"Rapport : {out.relative_to(compare.ROOT).as_posix()}")


if __name__ == "__main__":
    main()
