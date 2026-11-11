"""Mesures du biais de source du jeu Kaggle et critères S1 à S3 (étape 11b-6a).

Dans Kaggle, la taille d'image (512 x 512 ou autre) trahit la source, et donc presque la classe :
les gliomes sont quasi tous en 512 x 512, les images « notumor » sont presque toutes d'autres
tailles. Ce module fournit :

    - :func:`group_table` : exactitude par groupe classe x source, avec intervalle de confiance ;
    - :func:`group_weights` : poids par image donnant le même poids total à chaque groupe ;
    - :func:`source_criteria` : critères S1 à S3 et garde-fous, écrits AVANT mesure :

        - S1 : exactitude sur les tumeurs hors 512 du test >= 80 % ET intervalle de confiance de
          l'écart apparié avec A0 sans 0 (et positif) ;
        - S2 : au plus 5 tumeurs prédites « notumor » parmi les tumeurs du test ;
        - S3 : estimation hors pli sur les tumeurs hors 512 du train >= 80 %, borne basse de
          l'intervalle >= 75 % ;
        - garde-fous : rappel « notumor » >= 95 % et exactitude globale >= 87 %.
"""

from __future__ import annotations

import numpy as np

from brain_mri.model_selection import NOTUMOR, bootstrap_mean

S1_MIN_ACCURACY = 0.80  # exactitude minimale sur les tumeurs hors 512
S2_MAX_TUMOR_AS_NOTUMOR = 5  # tumeurs prédites « notumor » au plus
S3_MIN_ACCURACY = 0.80  # estimation hors pli minimale
S3_MIN_CI_LOW = 0.75  # borne basse minimale de l'intervalle de l'estimation hors pli
GUARD_MIN_NOTUMOR_RECALL = 0.95  # rappel « notumor » minimal
GUARD_MIN_ACCURACY = 0.87  # exactitude globale minimale
DEFAULT_MIN_COUNT = 30  # un groupe plus petit est pondéré comme s'il avait 30 images


def group_table(
    labels: np.ndarray, predictions: np.ndarray, is_512: np.ndarray, class_names: list[str]
) -> list[dict]:
    """Exactitude par groupe classe x source, avec IC, et taux de tumeurs prédites « notumor ».

    Args:
        labels: Vraies classes (indices), ``(N,)``.
        predictions: Classes prédites (indices), ``(N,)``.
        is_512: Booléens « l'image fait 512 x 512 », ``(N,)``.
        class_names: Noms des classes, dans l'ordre des indices.

    Returns:
        Une ligne par groupe non vide : ``class``, ``is_512``, ``n``, ``accuracy`` (sortie de
        :func:`bootstrap_mean`), ``predicted_notumor`` (nombre) et ``predicted_notumor_rate``
        (``None`` pour la classe « notumor » elle-même).
    """
    notumor = class_names.index(NOTUMOR)
    correct = predictions == labels
    rows = []
    for index, name in enumerate(class_names):
        for source in (True, False):
            mask = (labels == index) & (is_512 == source)
            if not mask.any():
                continue
            as_notumor = int((predictions[mask] == notumor).sum())
            is_tumor = index != notumor
            rows.append(
                {
                    "class": name,
                    "is_512": source,
                    "n": int(mask.sum()),
                    "accuracy": bootstrap_mean(correct[mask]),
                    "predicted_notumor": as_notumor if is_tumor else None,
                    "predicted_notumor_rate": as_notumor / int(mask.sum()) if is_tumor else None,
                }
            )
    return rows


def group_weights(
    labels: np.ndarray, is_512: np.ndarray, min_count: int = DEFAULT_MIN_COUNT
) -> np.ndarray:
    """Poids par image : chaque groupe classe x source pèse autant au total.

    Un groupe de ``n`` images reçoit le poids ``1 / max(n, min_count)`` par image : tous les
    groupes d'au moins ``min_count`` images ont donc un poids total de 1 ; un groupe plus
    petit est pondéré comme s'il avait ``min_count`` images (son poids total reste < 1), pour
    ne pas faire dominer quelques images rares.

    Args:
        labels: Vraies classes (indices), ``(N,)``.
        is_512: Booléens « l'image fait 512 x 512 », ``(N,)``.
        min_count: Effectif plancher d'un groupe.

    Returns:
        Poids ``(N,)`` en décimaux.
    """
    weights = np.zeros(len(labels), dtype=np.float64)
    for label in np.unique(labels):
        for source in (True, False):
            mask = (labels == label) & (is_512 == source)
            if mask.any():
                weights[mask] = 1.0 / max(int(mask.sum()), min_count)
    return weights


def source_criteria(
    labels: np.ndarray,
    predictions: np.ndarray,
    is_512: np.ndarray,
    class_names: list[str],
    reference_predictions: np.ndarray | None,
    oof_correct: np.ndarray | None,
) -> dict:
    """Applique S1 à S3 et les garde-fous à un modèle, sur le test.

    Args:
        labels: Vraies classes du test, ``(N,)``.
        predictions: Prédictions du modèle sur le test.
        is_512: Booléens « 512 x 512 » du test.
        class_names: Noms des classes.
        reference_predictions: Prédictions de A0 sur le test, ou ``None`` pour A0 lui-même
            (l'écart apparié n'existe pas : ``passed`` de S1 vaut alors ``None``).
        oof_correct: Booléens « bien classé hors pli » sur les tumeurs hors 512 du train,
            ou ``None`` si cette estimation n'existe pas encore (S3 : ``passed`` = ``None``).

    Returns:
        ``{"S1", "S2", "S3", "guards"}`` avec les valeurs utiles et ``passed``.
    """
    notumor = class_names.index(NOTUMOR)
    correct = predictions == labels
    target = (labels != notumor) & ~is_512  # tumeurs hors 512 du test
    acc = bootstrap_mean(correct[target])
    s1: dict = {"accuracy_tumors_not_512": acc}
    if reference_predictions is None:
        s1["passed"] = None
        s1["reason"] = "modèle de référence"
    else:
        ref_correct = reference_predictions == labels
        diff = bootstrap_mean(correct[target].astype(np.int64) - ref_correct[target])
        s1["diff_vs_a0"] = diff
        s1["passed"] = bool(
            acc["value"] is not None
            and acc["value"] >= S1_MIN_ACCURACY
            and diff["ci_low"] > 0  # IC sans 0, du côté d'un gain
        )
    tumor_as_notumor = int(((labels != notumor) & (predictions == notumor)).sum())
    s2 = {
        "tumor_predicted_notumor": tumor_as_notumor,
        "passed": tumor_as_notumor <= S2_MAX_TUMOR_AS_NOTUMOR,
    }
    if oof_correct is None:
        s3: dict = {"passed": None, "reason": "estimation hors pli absente"}
    else:
        est = bootstrap_mean(np.asarray(oof_correct))
        s3 = {
            "oof_accuracy": est,
            "passed": bool(
                est["value"] is not None
                and est["value"] >= S3_MIN_ACCURACY
                and est["ci_low"] >= S3_MIN_CI_LOW
            ),
        }
    recall = float(correct[labels == notumor].mean())
    accuracy = float(correct.mean())
    guards = {
        "notumor_recall": recall,
        "accuracy": accuracy,
        "passed": recall >= GUARD_MIN_NOTUMOR_RECALL and accuracy >= GUARD_MIN_ACCURACY,
    }
    return {"S1": s1, "S2": s2, "S3": s3, "guards": guards}
