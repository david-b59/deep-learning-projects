"""Mesure les modèles sur le test externe « intra-source » Pandit 2025 (étape 11b-7).

Données : coupes PNG de ``data/external/pandit/slices/`` décrites par le manifeste versionné
``ml/splits/external_pandit.csv`` (voir ``extract_external_slices.py`` et
``ml/splits/external_rules.md``). TEST SEULEMENT : aucun entraînement.

Par modèle ET par séquence (T1 avec contraste, T2) :
    - sur les coupes TUMEUR : part prédite « pituitary », « autre tumeur » (glioma ou
      meningioma) et « notumor » ;
    - sur les coupes SAINES : spécificité (part prédite « notumor »).

Critères fixés avant la mesure :
    E1 : coupes tumeur prédites « notumor » <= 10 % ;
    E2 : spécificité sur les coupes saines >= 80 %.
Intervalles de confiance à 95 % : ``bootstrap_mean`` (graine 42).

Chaque modèle utilise son propre prétraitement (configuration de son entraînement), comme
``evaluate.py``. Résultat : ``docs/figures/external_11b.json``.

Remarque : ``evaluate_external.py`` (étape 11a, jeu PMRAM) existe déjà et n'est pas modifié ;
ce script a donc un nom distinct.

Utilisation (depuis ``brain-tumor-classifier/ml/``) ::

    PYTHONUTF8=1 HF_HUB_OFFLINE=1 uv run python scripts/evaluate_external_pandit.py
    PYTHONUTF8=1 HF_HUB_OFFLINE=1 uv run python scripts/evaluate_external_pandit.py --with-b-dfr
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from brain_mri.config import load_config
from brain_mri.model import build_model_for_checkpoint, get_device, load_checkpoint
from brain_mri.model_selection import NOTUMOR, bootstrap_mean
from brain_mri.transforms import build_transforms

LOGGER = logging.getLogger("evaluate_external_pandit")
ROOT = Path(__file__).resolve().parents[2]  # brain-tumor-classifier/
MANIFEST = ROOT / "ml" / "splits" / "external_pandit.csv"
SLICES_DIR = ROOT / "data" / "external" / "pandit" / "slices"
OUTPUT = ROOT / "docs" / "figures" / "external_11b.json"
SEED = 42
E1_MAX = 0.10  # tumeurs prédites « notumor » : 10 % au plus
E2_MIN = 0.80  # spécificité sur les coupes saines : 80 % au moins
PITUITARY = "pituitary"
MODELS = {  # modèle : (point de contrôle, configuration d'entraînement = prétraitement)
    "a0": ("models/a0/best.pt", "ml/configs/train_a0.yaml"),
    "a1": ("models/a1/best.pt", "ml/configs/train_a1.yaml"),
    "b0": ("models/b0/best.pt", "ml/configs/train_b0.yaml"),
    "b1": ("models/b1/best.pt", "ml/configs/train_b1.yaml"),
    "online": ("models/backup/best_v1_online.pt", "ml/configs/train.yaml"),
    "b_dfr": ("models/b_dfr/best.pt", "ml/configs/train_b_dfr.yaml"),  # option --with-b-dfr
}
DEFAULT_MODELS = ("a0", "a1", "b0", "b1", "online")


def read_manifest() -> list[dict]:
    """Lit le manifeste du test externe.

    Returns:
        Une ligne (dictionnaire) par image.
    """
    with open(MANIFEST, encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def predict_files(
    model: torch.nn.Module, files: list[str], transform, device: torch.device, batch: int = 16
) -> tuple[np.ndarray, np.ndarray]:
    """Prédit la classe de chaque PNG.

    Args:
        model: Modèle en mode évaluation.
        files: Noms de fichiers dans ``SLICES_DIR``.
        transform: Prétraitement du modèle.
        device: CPU ou GPU.
        batch: Taille des lots.

    Returns:
        ``(classes prédites, probabilités)`` de formes ``(N,)`` et ``(N, classes)``.
    """
    preds: list[np.ndarray] = []
    probs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(files), batch):
            images = []
            for name in files[start : start + batch]:
                with Image.open(SLICES_DIR / name) as image:
                    images.append(transform(image.convert("RGB")))
            p = torch.softmax(model(torch.stack(images).to(device)), dim=1).cpu().numpy()
            probs.append(p)
            preds.append(p.argmax(axis=1))
    return np.concatenate(preds), np.concatenate(probs)


def summarize_sequence(rows: list[dict], preds: np.ndarray, class_names: list[str]) -> dict:
    """Résume les prédictions d'une séquence : tumeur (3 parts) et spécificité, avec IC.

    Args:
        rows: Lignes du manifeste de cette séquence (même ordre que ``preds``).
        preds: Classes prédites.
        class_names: Noms des classes du modèle.

    Returns:
        Dictionnaire avec ``tumor`` (pituitary, other_tumor, notumor), ``healthy``
        (specificity) et les critères E1 et E2.
    """
    names = np.array([class_names[p] for p in preds])
    tumor = np.array([r["label"] == "tumor" for r in rows])
    healthy = ~tumor
    result: dict = {"n_tumor": int(tumor.sum()), "n_healthy": int(healthy.sum())}
    result["tumor"] = {
        "pituitary": bootstrap_mean((names[tumor] == PITUITARY).astype(float), seed=SEED),
        "other_tumor": bootstrap_mean(
            ((names[tumor] != PITUITARY) & (names[tumor] != NOTUMOR)).astype(float), seed=SEED
        ),
        "notumor": bootstrap_mean((names[tumor] == NOTUMOR).astype(float), seed=SEED),
    }
    result["healthy"] = {
        "specificity": bootstrap_mean((names[healthy] == NOTUMOR).astype(float), seed=SEED),
        "predicted_pituitary": bootstrap_mean(
            (names[healthy] == PITUITARY).astype(float), seed=SEED
        ),
    }
    e1 = result["tumor"]["notumor"]["value"]
    e2 = result["healthy"]["specificity"]["value"]
    result["E1"] = {"threshold": E1_MAX, "value": e1, "passed": e1 is not None and e1 <= E1_MAX}
    result["E2"] = {"threshold": E2_MIN, "value": e2, "passed": e2 is not None and e2 >= E2_MIN}
    return result


def evaluate_model(name: str, rows: list[dict], device: torch.device) -> dict:
    """Mesure un modèle sur toutes les séquences du manifeste.

    Args:
        name: Nom du modèle (clé de ``MODELS``).
        rows: Lignes du manifeste.
        device: CPU ou GPU.

    Returns:
        ``{"class_names", "sequences": {séquence: résumé}}``.

    Raises:
        FileNotFoundError: Si le point de contrôle est absent.
    """
    checkpoint_rel, config_rel = MODELS[name]
    path = ROOT / checkpoint_rel
    if not path.is_file():
        raise FileNotFoundError(f"Modèle introuvable : {checkpoint_rel}")
    config = load_config(ROOT / config_rel)
    checkpoint = load_checkpoint(path, device)
    class_names = list(checkpoint["class_names"])
    model = build_model_for_checkpoint(checkpoint, config.model).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    transform = build_transforms(config.data, train=False)  # prétraitement propre au modèle
    report: dict = {"class_names": class_names, "sequences": {}}
    for sequence in sorted({r["sequence"] for r in rows}):
        subset = [r for r in rows if r["sequence"] == sequence]
        preds, _ = predict_files(model, [r["file"] for r in subset], transform, device)
        report["sequences"][sequence] = summarize_sequence(subset, preds, class_names)
    return report


def pct(value: float | None) -> str:
    """Formate une proportion en pourcentage français (« — » si absente)."""
    return "—" if value is None else f"{100 * value:.1f} %".replace(".", ",")


def print_table(report: dict) -> None:
    """Affiche en français le tableau modèle x séquence.

    Args:
        report: Rapport complet (clé ``models``).
    """
    print("\nmodèle   séq  | tumeurs : pituitary / autre / notumor (E1) | sain : spécificité (E2)")
    for name, block in report["models"].items():
        for sequence, res in block["sequences"].items():
            t, h = res["tumor"], res["healthy"]["specificity"]
            print(
                f"{name:<8} {sequence:<3}  | {pct(t['pituitary']['value'])} / "
                f"{pct(t['other_tumor']['value'])} / {pct(t['notumor']['value'])} "
                f"[{'OK' if res['E1']['passed'] else 'NON'}] | "
                f"{pct(h['value'])} [{'OK' if res['E2']['passed'] else 'NON'}]  "
                f"(n = {res['n_tumor']} tumeur, {res['n_healthy']} saines)"
            )


def main() -> None:
    """Point d'entrée : mesure les modèles et écrit ``external_11b.json``."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--with-b-dfr", action="store_true", help="ajoute models/b_dfr/best.pt")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    torch.manual_seed(0)
    rows = read_manifest()
    device = get_device()
    names = list(DEFAULT_MODELS) + (["b_dfr"] if args.with_b_dfr else [])
    report: dict = {
        "dataset": "Pandit 2025, doi 10.6084/m9.figshare.27894084.v1 (CC BY 4.0)",
        "criteria": {
            "E1": f"tumeurs prédites notumor <= {E1_MAX:.0%}",
            "E2": f"spécificité >= {E2_MIN:.0%}",
        },
        "n_patients": len({r["patient"] for r in rows}),
        "n_images": len(rows),
        "models": {},
    }
    for name in names:
        LOGGER.info("Modèle %s", name)
        report["models"][name] = evaluate_model(name, rows, device)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print_table(report)
    print(f"\nRapport enregistré dans {OUTPUT.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
