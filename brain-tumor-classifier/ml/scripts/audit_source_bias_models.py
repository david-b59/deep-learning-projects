"""Mesure le biais de source de A0, A1, B0 et B1 sur le test honnête (étape 11b-6a).

Trois mesures, écrites dans ``docs/figures/source_bias_11b.json`` et résumées en français :

    1. exactitude par groupe classe x source (512 x 512 ou autre) avec intervalle de confiance,
       et tumeurs prédites « notumor » par groupe ; critères S1, S2 et garde-fous (S3 attend
       l'estimation hors pli de l'étape 11b-6b) ;
    2. SONDE DE SOURCE : régression logistique (standardisation + LogisticRegression) qui
       prédit « 512 x 512 » à partir des caractéristiques, sur les TUMEURS seulement, apprise
       sur le train, AUC sur le test. Une AUC proche de 1 : la source est lisible dans les
       caractéristiques ;
    3. SONDE INTRA-SOURCE : tumeur (1) contre « notumor » (0), apprise seulement sur les images
       HORS 512 du train (``class_weight="balanced"``), puis rappel des tumeurs et spécificité
       sur les images hors 512 du test. Mesure si les caractéristiques savent séparer tumeur et
       sain à l'intérieur d'une même source.

Caractéristiques : 2048 avant ``fc`` pour A0 et A1 (``extract_penultimate.py``) ; pour B0 et B1
une seule sonde « dinov2 » sur la vue d'ÉVALUATION (même corps gelé, donc mêmes caractéristiques
de base ; B1 n'en diffère que par sa tête, entraînée sur 8 vues augmentées).

Les prédictions retenues sont reconstruites depuis ``docs/figures/<m>/evaluation.json`` (comme
``compare_models.py``). Elles sont aussi recalculées depuis les caractéristiques et la couche
finale du point de contrôle ; la concordance est écrite dans le JSON.

Utilisation (depuis ``brain-tumor-classifier/ml/``, après ``extract_penultimate.py``) ::

    uv run python scripts/audit_source_bias_models.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from brain_mri.config import load_config
from brain_mri.data import read_split_manifest
from brain_mri.dinov2_head import get_features
from brain_mri.model_selection import NOTUMOR, bootstrap_mean
from brain_mri.source_bias import group_table, source_criteria

ROOT = Path(__file__).resolve().parents[2]  # dossier brain-tumor-classifier/
FIGURES_DIR = ROOT / "docs" / "figures"
MANIFEST = ROOT / "ml" / "splits" / "kaggle_grouped.csv"
CACHE_DIR = ROOT / "data" / "cache" / "penultimate"
SEED = 42
MODELS = {  # modèle : (jeu de caractéristiques, point de contrôle, préfixe de la couche finale)
    "a0": ("a0", ROOT / "models" / "a0" / "best.pt", "fc"),
    "a1": ("a1", ROOT / "models" / "a1" / "best.pt", "fc"),
    "b0": ("dinov2", ROOT / "models" / "b0" / "best.pt", "head"),
    "b1": ("dinov2", ROOT / "models" / "b1" / "best.pt", "head"),
}
EXTRA_MODELS = {  # ajouté à l'étape 11b-6b : jamais dans source_bias_11b.json (voir report_dfr.py)
    "b_dfr": ("dinov2", ROOT / "models" / "b_dfr" / "best.pt", "head"),
}


def pct(value: float | None) -> str:
    """Formate une proportion en pourcentage français à une décimale (« — » si absente)."""
    return "—" if value is None else f"{100 * value:.1f} %".replace(".", ",")


def load_manifest() -> dict[str, dict]:
    """Lit le manifeste et regroupe, par partie, fichiers, classes et indicateur 512.

    Returns:
        ``{partie: {"files", "labels", "is_512"}}`` dans l'ordre du manifeste (celui des
        caractéristiques) ; ``class_names`` est sous la clé ``"class_names"`` de la partie
        ``train``.
    """
    with open(MANIFEST, encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    class_names = sorted({r["class"] for r in rows if r["part"] in ("train", "val", "test")})
    parts: dict[str, dict] = {}
    for part in ("train", "val", "test"):
        sub = [r for r in rows if r["part"] == part]
        parts[part] = {
            "files": [r["file"] for r in sub],
            "labels": np.array([class_names.index(r["class"]) for r in sub]),
            "is_512": np.array([r["is_512"] == "1" for r in sub]),
        }
    parts["train"]["class_names"] = class_names
    return parts


def load_features(feature_set: str, part: str) -> np.ndarray:
    """Charge les caractéristiques d'un jeu pour une partie.

    Args:
        feature_set: ``a0``, ``a1`` (cache ``penultimate``) ou ``dinov2`` (cache DINOv2).
        part: ``train``, ``val`` ou ``test``.

    Returns:
        Tableau ``(N, d)`` en décimaux, dans l'ordre du manifeste.
    """
    if feature_set != "dinov2":
        return torch.load(CACHE_DIR / f"{feature_set}_{part}.pt", weights_only=True)[
            "features"
        ].numpy()
    config = load_config(ROOT / "ml" / "configs" / "train_b0.yaml")
    samples, class_names = read_split_manifest(config.data)
    items = [(p, class_names.index(label)) for p, label in samples[part]]
    result = get_features(
        None, part, items, class_names, config.data, config.dinov2, 1, False, config.data.seed
    )
    return result.features[:, 0].numpy()


def reconstruct_predictions(
    name: str, files: list[str], labels: np.ndarray, class_names: list[str]
) -> tuple[np.ndarray, float]:
    """Reconstruit les prédictions d'un modèle depuis ``evaluation.json`` (liste d'erreurs).

    Args:
        name: Nom du modèle (dossier de ``docs/figures``).
        files: Images du test.
        labels: Vraies classes du test.
        class_names: Noms des classes.

    Returns:
        ``(prédictions, exactitude du fichier)``.

    Raises:
        ValueError: Si l'exactitude retrouvée diffère de celle du fichier.
    """
    report = json.loads((FIGURES_DIR / name / "evaluation.json").read_text(encoding="utf-8"))
    wrong = {e["file"]: class_names.index(e["predicted"]) for e in report["errors"]}
    predictions = np.array([wrong.get(f, y) for f, y in zip(files, labels, strict=True)])
    if abs(float((predictions == labels).mean()) - report["accuracy"]) > 1e-9:
        raise ValueError(f"{name} : exactitude reconstruite différente de evaluation.json")
    return predictions, report["accuracy"]


def recompute_predictions(checkpoint: Path, prefix: str, features: np.ndarray) -> np.ndarray:
    """Applique la couche finale d'un point de contrôle aux caractéristiques.

    Args:
        checkpoint: Fichier ``best.pt``.
        prefix: ``fc`` (ResNet) ou ``head`` (DINOv2).
        features: Caractéristiques ``(N, d)``.

    Returns:
        Classes prédites (indices).
    """
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)["state_dict"]
    logits = torch.from_numpy(features) @ state[f"{prefix}.weight"].T + state[f"{prefix}.bias"]
    return logits.argmax(dim=1).numpy()


def fit_probe(x: np.ndarray, y: np.ndarray, balanced: bool):
    """Entraîne une sonde : standardisation puis régression logistique (graine fixe).

    Args:
        x: Caractéristiques d'apprentissage.
        y: Cibles binaires.
        balanced: True pour ``class_weight="balanced"``.

    Returns:
        Le pipeline scikit-learn entraîné.
    """
    probe = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=5000, random_state=SEED, class_weight="balanced" if balanced else None
        ),
    )
    return probe.fit(x, y)


def run_probes(feature_set: str, data: dict[str, dict], class_names: list[str]) -> dict:
    """Sonde de source et sonde intra-source pour un jeu de caractéristiques.

    Args:
        feature_set: ``a0``, ``a1`` ou ``dinov2``.
        data: Sortie de :func:`load_manifest`.
        class_names: Noms des classes.

    Returns:
        ``{"source_probe": {...}, "within_source_probe": {...}}``.
    """
    notumor = class_names.index(NOTUMOR)
    tr, te = data["train"], data["test"]
    x_tr, x_te = load_features(feature_set, "train"), load_features(feature_set, "test")
    # Sonde de source : tumeurs seulement, cible = « 512 x 512 »
    tum_tr, tum_te = tr["labels"] != notumor, te["labels"] != notumor
    probe = fit_probe(x_tr[tum_tr], tr["is_512"][tum_tr], balanced=False)
    scores = probe.predict_proba(x_te[tum_te])[:, 1]
    auc = float(roc_auc_score(te["is_512"][tum_te], scores))
    source = {
        "n_train": int(tum_tr.sum()),
        "n_test": int(tum_te.sum()),
        "auc_test": auc,
        "accuracy_test": float((probe.predict(x_te[tum_te]) == te["is_512"][tum_te]).mean()),
    }
    # Sonde intra-source : tumeur (1) contre notumor (0), images hors 512 seulement
    out_tr, out_te = ~tr["is_512"], ~te["is_512"]
    y_tr = (tr["labels"][out_tr] != notumor).astype(int)
    probe = fit_probe(x_tr[out_tr], y_tr, balanced=True)
    y_te = (te["labels"][out_te] != notumor).astype(int)
    pred = probe.predict(x_te[out_te])
    within = {
        "n_train": int(out_tr.sum()),
        "n_train_tumors": int(y_tr.sum()),
        "recall_tumors": bootstrap_mean((pred[y_te == 1] == 1).astype(np.int64)),
        "specificity_notumor": bootstrap_mean((pred[y_te == 0] == 0).astype(np.int64)),
    }
    return {"source_probe": source, "within_source_probe": within}


def main() -> None:
    """Mesure les quatre modèles, lance les sondes, écrit le JSON et le résumé console."""
    data = load_manifest()
    class_names = data["train"]["class_names"]
    te = data["test"]
    labels, is_512 = te["labels"], te["is_512"]
    print(f"Test : {len(labels)} images, dont {int((~is_512).sum())} hors 512 x 512")
    predictions, models = {}, {}
    for name, (feature_set, checkpoint, prefix) in MODELS.items():
        predictions[name], file_accuracy = reconstruct_predictions(
            name, te["files"], labels, class_names
        )
        recomputed = recompute_predictions(checkpoint, prefix, load_features(feature_set, "test"))
        models[name] = {
            "concordance": {
                "accuracy_evaluation_json": file_accuracy,
                "accuracy_recomputed": float((recomputed == labels).mean()),
                "same_prediction_rate": float((recomputed == predictions[name]).mean()),
            }
        }
    for name in MODELS:
        reference = None if name == "a0" else predictions["a0"]
        models[name]["table"] = group_table(labels, predictions[name], is_512, class_names)
        models[name]["criteria"] = source_criteria(
            labels, predictions[name], is_512, class_names, reference, None
        )
        conc = models[name]["concordance"]
        print(
            f"\n== {name.upper()} : exactitude {pct(conc['accuracy_evaluation_json'])} "
            f"(recalculée {pct(conc['accuracy_recomputed'])}, mêmes prédictions "
            f"{pct(conc['same_prediction_rate'])})"
        )
        for row in models[name]["table"]:
            acc = row["accuracy"]
            extra = (
                f"  -> notumor : {row['predicted_notumor']}/{row['n']}"
                if row["predicted_notumor"] is not None
                else ""
            )
            print(
                f"   {row['class']:<11} {'512' if row['is_512'] else 'autre':<6} n={row['n']:<4} "
                f"exactitude {pct(acc['value'])} [{pct(acc['ci_low'])} ; {pct(acc['ci_high'])}]"
                f"{extra}"
            )
        crit = models[name]["criteria"]
        print(
            f"   S1 {crit['S1']['passed']} | S2 {crit['S2']['passed']} | "
            f"garde-fous {crit['guards']['passed']} | S3 attend l'estimation hors pli"
        )
    probes = {}
    for feature_set in ("a0", "a1", "dinov2"):
        probes[feature_set] = run_probes(feature_set, data, class_names)
        s, w = probes[feature_set]["source_probe"], probes[feature_set]["within_source_probe"]
        print(
            f"\nSonde [{feature_set}] source (tumeurs) : AUC test {s['auc_test']:.3f} "
            f"(n test {s['n_test']})\n   intra-source (hors 512) : rappel tumeurs "
            f"{pct(w['recall_tumors']['value'])} [{pct(w['recall_tumors']['ci_low'])} ; "
            f"{pct(w['recall_tumors']['ci_high'])}] (n {w['recall_tumors']['n']}), spécificité "
            f"{pct(w['specificity_notumor']['value'])} "
            f"[{pct(w['specificity_notumor']['ci_low'])} ; "
            f"{pct(w['specificity_notumor']['ci_high'])}] (n {w['specificity_notumor']['n']})"
        )
    report = {
        "test": {"n": len(labels), "n_not_512": int((~is_512).sum())},
        "note_probes": "B0 et B1 partagent la sonde « dinov2 » (vue d'évaluation, même corps)",
        "models": models,
        "probes": probes,
        "note_s3": "S3 (estimation hors pli) sera mesurée à l'étape 11b-6b",
    }
    out = FIGURES_DIR / "source_bias_11b.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nRapport : {out.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
