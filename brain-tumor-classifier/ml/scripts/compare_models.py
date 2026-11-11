"""Compare les quatre modèles de l'étape 11b et applique la règle de décision (11b-5).

Modèles comparés (résultats déjà produits par ``brain_mri.evaluate`` et
``scripts/evaluate_invariance.py``, dans ``docs/figures/<nom>/``) :

    - A0 : ResNet-50, recette du modèle en ligne, sur le découpage honnête ;
    - A1 : A0 + recadrage sur la tête et augmentation de cadrage ;
    - B0 : DINOv2-small gelé + couche linéaire, 1 vue par image ;
    - B1 : idem avec 8 vues augmentées par image d'entraînement.

Les prédictions de chaque modèle sont reconstruites à partir de ``evaluation.json`` (liste des
erreurs) et du manifeste ``ml/splits/kaggle_grouped.csv`` (partie ``test``) ; on vérifie que
l'exactitude retrouvée est exactement celle du fichier. Les critères C1 à C5 et la règle de choix
sont dans :mod:`brain_mri.model_selection`.

Sortie : ``docs/figures/model_comparison_11b.json``.

Utilisation (depuis ``brain-tumor-classifier/ml/``) ::

    uv run python scripts/compare_models.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from brain_mri.model_selection import bootstrap_mean, criteria, decide, model_metrics

ROOT = Path(__file__).resolve().parents[2]  # dossier brain-tumor-classifier/
FIGURES_DIR = ROOT / "docs" / "figures"
MANIFEST = ROOT / "ml" / "splits" / "kaggle_grouped.csv"
STRICT_COSINE = 0.94  # sous-test strict : aucune image d'entraînement à cosinus >= 0,94
REFERENCE = "a0"  # modèle de référence des critères C1 et C4
MODELS = {  # nom : (description, fichier d'invariance)
    "a0": ("ResNet-50, recette actuelle", "invariance_legacy.json"),
    "a1": ("ResNet-50 + recadrage sur la tête", "invariance_head_crop.json"),
    "b0": ("DINOv2-small gelé, 1 vue", "invariance_head_crop.json"),
    "b1": ("DINOv2-small gelé, 8 vues augmentées", "invariance_head_crop.json"),
}


def pct(value: float) -> str:
    """Formate une proportion en pourcentage français à une décimale."""
    return f"{100 * value:.1f} %".replace(".", ",")


def load_test_set() -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Lit les images de test du manifeste.

    Returns:
        ``(fichiers, classes, is_512, strict, noms_des_classes)`` ; ``strict`` marque les images
        sans voisine d'entraînement à cosinus >= 0,94.
    """
    with open(MANIFEST, encoding="utf-8", newline="") as file:
        rows = [row for row in csv.DictReader(file) if row["part"] == "test"]
    class_names = sorted({row["class"] for row in rows})
    files = [row["file"] for row in rows]
    labels = np.array([class_names.index(row["class"]) for row in rows])
    is_512 = np.array([row["is_512"] == "1" for row in rows])
    strict = np.array([float(row["max_cos_to_train"]) < STRICT_COSINE for row in rows])
    return files, labels, is_512, strict, class_names


def load_predictions(
    name: str, files: list[str], labels: np.ndarray, class_names: list[str]
) -> np.ndarray:
    """Reconstruit les prédictions d'un modèle à partir de sa liste d'erreurs.

    Args:
        name: Nom du modèle (dossier dans ``docs/figures``).
        files: Images de test (chemins relatifs à ``data/raw``).
        labels: Vraies classes des images.
        class_names: Noms des classes.

    Returns:
        Les classes prédites (indices), dans l'ordre de ``files``.

    Raises:
        ValueError: Si l'exactitude retrouvée diffère de celle de ``evaluation.json``.
    """
    report = json.loads((FIGURES_DIR / name / "evaluation.json").read_text(encoding="utf-8"))
    wrong = {e["file"]: class_names.index(e["predicted"]) for e in report["errors"]}
    predictions = np.array([wrong.get(f, label) for f, label in zip(files, labels, strict=True)])
    accuracy = float((predictions == labels).mean())
    if abs(accuracy - report["accuracy"]) > 1e-9:
        raise ValueError(f"{name} : exactitude {accuracy} au lieu de {report['accuracy']}")
    return predictions


def main() -> None:
    """Mesure chaque modèle, applique les critères et la règle, écrit le JSON."""
    files, labels, is_512, strict, class_names = load_test_set()
    print(f"Test : {len(files)} images, dont {int(strict.sum())} dans le sous-test strict")
    predictions = {n: load_predictions(n, files, labels, class_names) for n in MODELS}
    metrics = {
        n: model_metrics(labels, p, class_names, is_512, strict) for n, p in predictions.items()
    }
    reference_correct = predictions[REFERENCE] == labels

    models: dict[str, dict] = {}
    for name, (description, invariance_file) in MODELS.items():
        invariance = json.loads((FIGURES_DIR / name / invariance_file).read_text(encoding="utf-8"))
        diff = None
        if name != REFERENCE:  # écart apparié : mêmes images, juste(modèle) - juste(A0)
            diff = bootstrap_mean(
                (predictions[name] == labels).astype(np.int64) - reference_correct
            )
        checks = criteria(metrics[name], invariance, metrics[REFERENCE], diff)
        models[name] = {
            "description": description,
            "metrics": metrics[name],
            "diff_vs_a0": diff,
            "criteria": checks,
            "accuracy": metrics[name]["accuracy"]["value"],
            "pmram_flip_rate": checks["C2"]["pmram_flip_rate"],
        }
        status = " ".join(
            f"{c}:{'—' if v['passed'] is None else ('OK' if v['passed'] else 'non')}"
            for c, v in checks.items()
        )
        acc = metrics[name]["accuracy"]
        print(
            f"{name.upper()} {description:<38} exactitude {pct(acc['value'])} "
            f"[{pct(acc['ci_low'])} ; {pct(acc['ci_high'])}]  {status}"
        )
        if checks["C3"]["failed"]:
            print(f"     C3 raté sur : {', '.join(checks['C3']['failed'])}")

    decision = decide(models)
    online = json.loads((FIGURES_DIR / "reference_honest.json").read_text(encoding="utf-8"))
    report = {
        "test": {"n": len(files), "n_strict": int(strict.sum()), "strict_cosine": STRICT_COSINE},
        "reference_model": REFERENCE,
        "online_model_honest_reference": online["without_exact_duplicates"]["without_twin"][
            "overall"
        ],
        "models": models,
        "decision": decision,
        "note_c6": "C6 (parité navigateur, taille ONNX) se mesure seulement pour un gagnant",
    }
    out = FIGURES_DIR / "model_comparison_11b.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Décision : {decision['decision']} (gagnant : {decision['winner']})")
    print(f"Rapport : {out.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
