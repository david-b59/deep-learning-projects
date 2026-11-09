"""Exactitude « honnête » du modèle actuel, avec intervalle de confiance (étape 11b-1).

Le 94,4 % publié est gonflé : 77,6 % des images de ``Testing`` ont une jumelle (cosinus DINOv2
>= 0,94) dans ``Training``. La référence honnête est l'exactitude de ``models/best.pt`` sur les
images de ``Testing`` SANS jumelle, mesurée avec le prétraitement du test
(``build_transforms(train=False)``). Deux versions :

    - ``all_testing`` : les 1 600 images de ``Testing`` (attendu : 358 sans jumelle, 80,4 %) ;
    - ``without_exact_duplicates`` : sans les doublons exacts, comme ``evaluate.py``
      (attendu : 347 sans jumelle, 82,7 %).

Chaque exactitude est donnée avec un intervalle de confiance à 95 % par bootstrap (2 000
tirages, graine 42), globalement et par classe. Sortie : ``docs/figures/reference_honest.json``.

Utilisation (depuis ``brain-tumor-classifier/ml/``) ::

    uv run python scripts/evaluate_current_reference.py
"""

from __future__ import annotations

import json
import time

import numpy as np

from brain_mri.config import ModelConfig, load_config
from brain_mri.duplicates import load_or_create_report
from brain_mri.embeddings import embed_files, load_model
from brain_mri.leakage import bootstrap_accuracy, predict_paths
from brain_mri.model import build_model, get_device, load_checkpoint
from brain_mri.near_duplicates import DUPLICATE_COSINE, list_images, nearest_cosine
from brain_mri.transforms import build_transforms

DRAWS = 2000  # tirages bootstrap
SEED = 42  # graine du bootstrap


def pct(value: float | None) -> str:
    """Formate une proportion en pourcentage français (``n/a`` si absente)."""
    return "n/a" if value is None else f"{100 * value:.1f} %".replace(".", ",")


def summarize(correct: np.ndarray, labels: np.ndarray, mask: np.ndarray, names: list[str]) -> dict:
    """Exactitude avec IC, globale et par classe, sur les images retenues par ``mask``.

    Args:
        correct: Booléens « le modèle a juste », ``(N,)``.
        labels: Classe (indice) de chaque image.
        mask: Booléens « l'image est retenue ».
        names: Noms des classes.

    Returns:
        ``{"overall": {...}, "per_class": {classe: {...}}}`` (sorties de ``bootstrap_accuracy``).
    """
    return {
        "overall": bootstrap_accuracy(correct[mask], DRAWS, SEED),
        "per_class": {
            name: bootstrap_accuracy(correct[mask & (labels == i)], DRAWS, SEED)
            for i, name in enumerate(names)
        },
    }


def main() -> None:
    """Mesure le modèle actuel sur les images de test sans jumelle et écrit le JSON."""
    start = time.time()
    config = load_config()
    path = config.training.checkpoint_path
    if not path.is_file():
        raise FileNotFoundError(f"Modèle introuvable : {path}")
    cache = config.data.raw_dir.parent / "cache"
    device = get_device()
    checkpoint = load_checkpoint(path, device)
    names = list(checkpoint["class_names"])
    model = build_model(len(names), ModelConfig(pretrained=False, freeze_backbone=False))
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    transform = build_transforms(config.data, train=False)  # identique au test Kaggle

    train_paths = list_images(config.data.raw_dir / "Training")
    test_paths = list_images(config.data.raw_dir / "Testing")
    index = {name: i for i, name in enumerate(names)}
    labels = np.array([index[p.parent.name] for p in test_paths])
    dino = load_model()
    train_emb = embed_files(dino, train_paths, cache / "embeddings")
    test_emb = embed_files(dino, test_paths, cache / "embeddings")
    del dino
    twin, cosine = nearest_cosine(test_emb, train_emb)
    has_twin = cosine >= DUPLICATE_COSINE
    predictions = predict_paths(model, test_paths, transform, device, config.data.batch_size)
    correct = predictions == labels
    duplicate_report = load_or_create_report(config.data.raw_dir, config.data.processed_dir)
    excluded = duplicate_report.files_to_exclude()
    kept = np.array(
        [p.relative_to(config.data.raw_dir).as_posix() not in excluded for p in test_paths]
    )

    report = {
        "model": str(path.name),
        "duplicate_cosine": DUPLICATE_COSINE,
        "bootstrap": {"draws": DRAWS, "seed": SEED, "level": 0.95, "method": "percentiles"},
        "all_testing": {
            "n_images": len(labels),
            "accuracy_all_images": float(correct.mean()),
            "without_twin": summarize(correct, labels, ~has_twin, names),
            "with_twin": summarize(correct, labels, has_twin, names),
        },
        "without_exact_duplicates": {
            "n_images": int(kept.sum()),
            "accuracy_all_images": float(correct[kept].mean()),
            "without_twin": summarize(correct, labels, kept & ~has_twin, names),
            "with_twin": summarize(correct, labels, kept & has_twin, names),
        },
    }
    for key, title in (
        ("all_testing", "Testing complet"),
        ("without_exact_duplicates", "Testing sans doublons exacts"),
    ):
        block = report[key]
        for name in ("without_twin", "with_twin"):
            o = block[name]["overall"]
            print(
                f"{title} - {name} : {o['n']} images, exactitude {pct(o['accuracy'])} "
                f"[{pct(o['ci_low'])} ; {pct(o['ci_high'])}]"
            )
        for cls, o in block["without_twin"]["per_class"].items():
            print(
                f"    {cls:<11} sans jumelle : {o['n']:>4} images, {pct(o['accuracy'])} "
                f"[{pct(o['ci_low'])} ; {pct(o['ci_high'])}]"
            )
    out = config.training.figures_dir / "reference_honest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Rapport enregistré dans {out} ({time.time() - start:.0f} s)")


if __name__ == "__main__":
    main()
