"""Ajuste les garde-fous du modèle A1 (étape 11c-1), selon ``ml/splits/guards_rules.md``.

AUCUNE mesure sur le test Kaggle, le test Pandit ou le jeu hors sujet : ces données sont
interdites à l'ajustement (assertion explicite : :func:`brain_mri.guards.assert_no_forbidden_data`).

Étapes :
    1. relit les caractéristiques A1 en cache (``data/cache/penultimate/a1_{train,val}.pt``) et
       vérifie qu'elles correspondent au manifeste (ordre, étiquettes, recalcul d'un échantillon) ;
    2. extrait (une seule fois) celles des 320 coupes Pandit développement, avec EXACTEMENT le
       prétraitement ``head_crop`` d'A1 ;
    3. couleur : statistique sur les images natives de Kaggle train + val, seuil = 2 x le maximum ;
    4. température (LBFGS, NLL de la validation), ECE avant / après ;
    5. ACP 256 + LedoitWolf + 4 centres sur le train, tau_D = 97e percentile de la validation ;
    6. tau_P : plus grande valeur gardant la part refusée de la validation <= 5 % ;
    7. détecteur T2 : régression logistique L2 sur Pandit développement (C par validation croisée
       groupée par patient, 5 plis), tau_T2 selon la règle (figshare val et T1c Pandit hors pli) ;
    8. écrit ``models/a1/guards.safetensors`` et ``docs/figures/guards_fit_11c.json``.

Utilisation (depuis ``brain-tumor-classifier/``) ::

    PYTHONUTF8=1 ml/.venv/Scripts/python ml/scripts/fit_guards.py
"""

from __future__ import annotations

import csv
import logging
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader

from brain_mri import guards
from brain_mri.config import ModelConfig, load_config
from brain_mri.data import ManifestDataset, read_split_manifest
from brain_mri.model import build_model, get_device, load_checkpoint
from brain_mri.transforms import build_transforms

LOGGER = logging.getLogger("fit_guards")
ROOT = Path(__file__).resolve().parents[2]  # dossier brain-tumor-classifier/
CONFIG = ROOT / "ml" / "configs" / "train_a1.yaml"
CHECKPOINT = ROOT / "models" / "a1" / "best.pt"
CACHE_DIR = ROOT / "data" / "cache" / "penultimate"
RAW_DIR = ROOT / "data" / "raw"
KAGGLE_MANIFEST = ROOT / "ml" / "splits" / "kaggle_grouped.csv"
PANDIT_DEV_MANIFEST = ROOT / "ml" / "splits" / "pandit_dev.csv"
PANDIT_TEST_MANIFEST = ROOT / "ml" / "splits" / "external_pandit.csv"
OFFTOPIC_MANIFEST = ROOT / "ml" / "splits" / "offtopic.csv"
PANDIT_DEV_DIR = ROOT / "data" / "external" / "pandit_dev" / "slices"
OUT_WEIGHTS = ROOT / "models" / "a1" / "guards.safetensors"
OUT_JSON = ROOT / "docs" / "figures" / "guards_fit_11c.json"
DISCARDED_TEST_PATIENT = "16"  # tiré à l'étape 11b-7 mais écarté (masque hors grille) : interdit
C_GRID = (0.0001, 0.001, 0.01, 0.1, 1.0, 10.0)  # grille de la régression logistique
CV_SPLITS = 5
SEED = 42
MIN_FREE_GB = 2.0


def relative(path: Path) -> str:
    """Donne un chemin relatif à la racine du projet, avec des barres obliques.

    Args:
        path: Chemin absolu.

    Returns:
        Le chemin relatif (pour les JSON).
    """
    return path.resolve().relative_to(ROOT).as_posix()


def read_csv(path: Path) -> list[dict]:
    """Lit un CSV en liste de dictionnaires.

    Args:
        path: Fichier CSV.

    Returns:
        Une ligne par dictionnaire.
    """
    with open(path, encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def check_disk() -> None:
    """Arrête le script si le disque a moins de 2 Go libres.

    Raises:
        SystemExit: Si la place libre est insuffisante.
    """
    free = shutil.disk_usage(ROOT).free / 1e9
    if free < MIN_FREE_GB:
        raise SystemExit(f"Disque presque plein ({free:.1f} Go libres) : arrêt.")


def load_a1(device: torch.device) -> tuple[nn.Module, list[str]]:
    """Charge A1 complet (avec sa couche fc) et ses noms de classes.

    Args:
        device: Appareil de calcul.

    Returns:
        ``(modèle en évaluation, noms de classes)``.
    """
    checkpoint = load_checkpoint(CHECKPOINT, device)
    class_names = list(checkpoint["class_names"])
    model = build_model(len(class_names), ModelConfig(pretrained=False, freeze_backbone=False))
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval(), class_names


def extract_features(
    model: nn.Module, items: list[tuple[Path, int]], class_names: list[str], device: torch.device
) -> torch.Tensor:
    """Calcule les 2048 caractéristiques avant ``fc`` avec la transformation d'ÉVALUATION d'A1.

    Args:
        model: A1 avec sa couche ``fc`` (elle est remplacée temporairement par l'identité).
        items: Couples ``(chemin, indice de classe)``.
        class_names: Noms de classes.
        device: Appareil de calcul.

    Returns:
        Un tenseur ``(N, 2048)`` sur CPU.
    """
    config = load_config(CONFIG)
    transform = build_transforms(config.data, train=False)  # head_crop, sans augmentation
    loader = DataLoader(ManifestDataset(items, transform, class_names), batch_size=32)
    fc = model.fc
    model.fc = nn.Identity()
    outputs = []
    try:
        with torch.no_grad():
            for images, _ in loader:
                outputs.append(model(images.to(device)).cpu())
    finally:
        model.fc = fc  # on rend le modèle intact
    return torch.cat(outputs)


def load_cached_features(
    model: nn.Module, class_names: list[str], device: torch.device
) -> dict[str, tuple[np.ndarray, np.ndarray, list[Path]]]:
    """Relit les caches train / val et vérifie qu'ils correspondent au manifeste.

    Contrôles : même nombre d'images, mêmes étiquettes dans le même ordre, et recalcul de 8
    images par partie (écart maximal toléré 1e-3).

    Args:
        model: A1 complet (pour le recalcul de contrôle).
        class_names: Noms de classes.
        device: Appareil de calcul.

    Returns:
        ``{partie: (caractéristiques, étiquettes, chemins)}`` pour ``train`` et ``val``.

    Raises:
        RuntimeError: Si un cache ne correspond pas au manifeste.
    """
    samples, _ = read_split_manifest(load_config(CONFIG).data)
    index = {c: i for i, c in enumerate(class_names)}
    result = {}
    for part in ("train", "val"):
        cache = torch.load(CACHE_DIR / f"a1_{part}.pt", weights_only=True)
        items = [(path, index[label]) for path, label in samples[part]]
        labels = np.array([t for _, t in items])
        if len(items) != len(cache["features"]) or not np.array_equal(
            labels, cache["labels"].numpy()
        ):
            raise RuntimeError(f"Le cache a1_{part}.pt ne correspond pas au manifeste")
        picks = np.random.default_rng(SEED).choice(len(items), size=8, replace=False)
        fresh = extract_features(model, [items[i] for i in picks], class_names, device)
        gap = (fresh - cache["features"][picks]).abs().max().item()
        if gap > 1e-3:
            raise RuntimeError(f"Cache {part} : écart de recalcul {gap:.2e} > 1e-3")
        LOGGER.info(
            "cache a1_%s : %d images conformes au manifeste (écart %.1e)", part, len(items), gap
        )
        result[part] = (cache["features"].numpy(), labels, [p for p, _ in items])
    return result


def pandit_dev_features(
    model: nn.Module, class_names: list[str], device: torch.device
) -> tuple[np.ndarray, list[dict]]:
    """Calcule (ou relit) les caractéristiques A1 des 320 coupes Pandit développement.

    Args:
        model: A1 complet.
        class_names: Noms de classes (la classe est factice : seules les images comptent).
        device: Appareil de calcul.

    Returns:
        ``(caractéristiques (320, 2048), lignes du manifeste dans le même ordre)``.
    """
    rows = read_csv(PANDIT_DEV_MANIFEST)
    target = CACHE_DIR / "a1_pandit_dev.pt"
    if target.is_file():
        cache = torch.load(target, weights_only=True)
        if cache["files"] == [r["file"] for r in rows]:
            return cache["features"].numpy(), rows
    items = [(PANDIT_DEV_DIR / r["file"], 0) for r in rows]
    features = extract_features(model, items, class_names, device)
    tmp = target.with_suffix(".tmp")
    torch.save({"features": features, "files": [r["file"] for r in rows]}, tmp)
    tmp.replace(target)
    LOGGER.info("Pandit dev : %s caractéristiques extraites", tuple(features.shape))
    return features.numpy(), rows


def kaggle_color_stats(paths: list[Path]) -> np.ndarray:
    """Calcule la statistique couleur de chaque image NATIVE (avant tout recadrage).

    Args:
        paths: Chemins des images Kaggle (train et val seulement).

    Returns:
        Un tableau de statistiques, une par image.
    """
    values = []
    for path in paths:
        with Image.open(path) as image:
            # Les images en niveaux de gris (mode L) valent 0 ; la palette et les autres modes
            # passent par RGB, ce qui reproduit ce que le navigateur verra en pixels.
            array = np.asarray(image if image.mode == "L" else image.convert("RGB"))
        values.append(guards.color_statistic(array))
    return np.array(values)


def fit_t2_detector(
    features: np.ndarray, rows: list[dict]
) -> tuple[float, dict, np.ndarray, tuple[np.ndarray, float]]:
    """Choisit C par validation croisée groupée, puis ajuste le détecteur T2 final.

    Régression logistique L2 (caractéristiques standardisées), cible = T2 (1) contre T1 (0),
    5 plis groupés par patient (``StratifiedGroupKFold``), critère = exactitude équilibrée
    à 0,5 sur les prédictions hors pli. Ensuite, refit sur tout le développement avec le C choisi.

    Args:
        features: Caractéristiques A1 de Pandit développement, ``(320, 2048)``.
        rows: Lignes du manifeste (``patient``, ``sequence``).

    Returns:
        ``(C choisi, exactitude équilibrée par C, probabilités hors pli au C choisi,
        (poids bruts, biais) du modèle final)``.
    """
    y = np.array([1 if r["sequence"] == "T2" else 0 for r in rows])
    groups = np.array([r["patient"] for r in rows])
    folds = list(
        StratifiedGroupKFold(CV_SPLITS, shuffle=True, random_state=SEED).split(features, y, groups)
    )
    scores: dict[str, float] = {}
    oof_by_c: dict[float, np.ndarray] = {}
    for c in C_GRID:
        oof = np.zeros(len(y))
        for train, test in folds:
            scaler = StandardScaler().fit(features[train])
            clf = LogisticRegression(C=c, max_iter=2000).fit(
                scaler.transform(features[train]), y[train]
            )
            oof[test] = clf.predict_proba(scaler.transform(features[test]))[:, 1]
        oof_by_c[c] = oof
        scores[str(c)] = float(balanced_accuracy_score(y, oof >= 0.5))
        LOGGER.info("C=%g : exactitude équilibrée hors pli %.4f", c, scores[str(c)])
    best_c = max(C_GRID, key=lambda c: (scores[str(c)], -c))  # égalité : le plus régularisé
    scaler = StandardScaler().fit(features)
    clf = LogisticRegression(C=best_c, max_iter=2000).fit(scaler.transform(features), y)
    final = guards.fold_standardization(
        clf.coef_[0], float(clf.intercept_[0]), scaler.mean_, scaler.scale_
    )
    return best_c, scores, oof_by_c[best_c], final


def main() -> None:
    """Point d'entrée : ajuste tous les garde-fous et écrit les deux fichiers de sortie."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    os.chdir(ROOT)  # les chemins de la configuration sont relatifs à la racine du projet
    check_disk()
    device = get_device()
    model, class_names = load_a1(device)
    cached = load_cached_features(model, class_names, device)
    x_train, y_train, _ = cached["train"]
    x_val, y_val, val_paths = cached["val"]
    train_paths = cached["train"][2]

    # --- Assertion d'interdiction (avant tout ajustement) -------------------------------------
    kaggle_rows = read_csv(KAGGLE_MANIFEST)
    by_file = {r["file"]: r for r in kaggle_rows}
    used_rows = [by_file[p.relative_to(RAW_DIR).as_posix()] for p in train_paths + val_paths]
    pandit_features, pandit_rows = pandit_dev_features(model, class_names, device)
    test_patients = {r["patient"] for r in read_csv(PANDIT_TEST_MANIFEST)} | {
        DISCARDED_TEST_PATIENT
    }
    offtopic_sha = {r["sha256"] for r in read_csv(OFFTOPIC_MANIFEST)}
    used_sha = {r["sha256"] for r in used_rows} | {r["sha256"] for r in pandit_rows}
    guards.assert_no_forbidden_data(
        used_rows,
        {r["patient"] for r in pandit_rows},
        test_patients,
        # Parties prévues ; le vrai contrôle porte sur la colonne « part » des lignes.
        used_kaggle_parts={"train", "val"},
        used_offtopic=used_sha & offtopic_sha,  # images utilisées présentes dans le hors sujet
    )
    LOGGER.info("Assertion OK : ni test Kaggle, ni patients du test Pandit, ni hors sujet")

    # --- Couleur ---------------------------------------------------------------------------------
    color_values = kaggle_color_stats(train_paths + val_paths)
    color_max = float(color_values.max())
    color_threshold = guards.color_threshold(color_max)
    LOGGER.info("couleur : max observé %.5f, seuil %.5f", color_max, color_threshold)

    # --- Température ---
    fc = model.fc.cpu()
    with torch.no_grad():
        val_logits = fc(torch.as_tensor(x_val)).numpy().astype(np.float64)
    temperature = guards.fit_temperature(val_logits, y_val)
    probs_before = torch.softmax(torch.as_tensor(val_logits), dim=1).numpy()
    probs_after = torch.softmax(torch.as_tensor(val_logits) / temperature, dim=1).numpy()
    ece_before = guards.expected_calibration_error(probs_before, y_val)
    ece_after = guards.expected_calibration_error(probs_after, y_val)
    accuracy = float((val_logits.argmax(1) == y_val).mean())
    LOGGER.info(
        "T=%.4f, ECE val %.4f -> %.4f (exactitude val %.4f)",
        temperature,
        ece_before,
        ece_after,
        accuracy,
    )

    # --- Mahalanobis, tau_D, tau_P ------------------------------------------------------------
    maha = guards.fit_mahalanobis(x_train, y_train)
    head = guards.GuardHead(x_train.shape[1], guards.PCA_COMPONENTS, len(class_names))
    with torch.no_grad():
        head.temperature.copy_(torch.tensor(temperature))
        head.pca_mean.copy_(torch.from_numpy(maha.mean))
        head.pca_components.copy_(torch.from_numpy(maha.components))
        head.precision.copy_(torch.from_numpy(maha.precision))
        head.centers.copy_(torch.from_numpy(maha.centers))
    head.eval()
    with torch.no_grad():
        val_probs, val_dist, _ = head(
            torch.as_tensor(val_logits, dtype=torch.float32), torch.as_tensor(x_val)
        )
    val_dist = val_dist.numpy()
    val_max = val_probs.numpy().max(axis=1)
    tau_d = guards.choose_tau_d(val_dist)
    tau_p = guards.choose_tau_p(val_dist, val_max, tau_d)
    g0 = guards.rejected_fraction(val_dist, val_max, tau_d, tau_p)
    LOGGER.info("tau_D=%.4f, tau_P=%.6f, part val refusée (G0) = %.4f", tau_d, tau_p, g0)

    # --- Détecteur T2 ----------------------------------------------------------------------------
    best_c, cv_scores, oof, (t2_weight, t2_bias) = fit_t2_detector(pandit_features, pandit_rows)
    with torch.no_grad():
        head.t2_weight.copy_(torch.from_numpy(t2_weight.astype(np.float32)))
        head.t2_bias.copy_(torch.tensor(t2_bias))
        _, _, p_val_t2 = head(
            torch.as_tensor(val_logits, dtype=torch.float32), torch.as_tensor(x_val)
        )
    # Images figshare de la validation : is_512 == 1 et classe != notumor (proxy documenté dans
    # le JSON, clé « choix_figshare »).
    val_rows = used_rows[len(train_paths) :]
    figshare = np.array([r["is_512"] == "1" and r["class"] != "notumor" for r in val_rows])
    figshare_prob = p_val_t2.numpy()[figshare]
    is_t2 = np.array([r["sequence"] == "T2" for r in pandit_rows])
    t1_oof = oof[~is_t2]
    tau_t2 = guards.choose_tau_t2(figshare_prob, t1_oof)
    fa_figshare = float((figshare_prob > tau_t2).mean())
    fa_pandit = float((t1_oof > tau_t2).mean())
    t2_recall_oof = float((oof[is_t2] > tau_t2).mean())
    cv_balanced = cv_scores[str(best_c)]
    LOGGER.info(
        "C=%g, tau_T2=%.4f, fausses alertes figshare val %.4f (n=%d), Pandit T1c hors pli %.4f, "
        "rappel T2 hors pli %.4f, exactitude équilibrée CV %.4f",
        best_c,
        tau_t2,
        fa_figshare,
        int(figshare.sum()),
        fa_pandit,
        t2_recall_oof,
        cv_balanced,
    )

    # --- Sauvegarde ------------------------------------------------------------------------------
    thresholds = guards.GuardThresholds(color_threshold, tau_d, tau_p, tau_t2)
    with torch.no_grad():
        head.thresholds.copy_(
            torch.tensor([thresholds.color, thresholds.tau_d, thresholds.tau_p, thresholds.tau_t2])
        )
    guards.save_guards(head, OUT_WEIGHTS)
    check = guards.load_guards(OUT_WEIGHTS)  # relecture de contrôle
    with torch.no_grad():
        again = check(torch.as_tensor(val_logits, dtype=torch.float32), torch.as_tensor(x_val))
    assert torch.allclose(again[1], torch.from_numpy(val_dist), atol=1e-4), "relecture incohérente"
    counts = {
        "kaggle_train": len(train_paths),
        "kaggle_val": len(val_paths),
        "kaggle_val_figshare": int(figshare.sum()),
        "pandit_dev_images": len(pandit_rows),
        "pandit_dev_patients": len({r["patient"] for r in pandit_rows}),
        "pandit_dev_t1": int((~is_t2).sum()),
        "pandit_dev_t2": int(is_t2.sum()),
    }
    guards.write_json(
        {
            "regles": "ml/splits/guards_rules.md",
            "modele": relative(CHECKPOINT),
            "poids_garde_fous": relative(OUT_WEIGHTS),
            "seuils": guards.thresholds_dict(thresholds),
            "couleur": {
                "maximum_observe_kaggle_train_val": color_max,
                "plancher": guards.COLOR_FLOOR,
            },
            "temperature": temperature,
            "ece_validation": {"avant": ece_before, "apres": ece_after},
            "exactitude_validation": accuracy,
            "g0_part_validation_refusee_distance_ou_confiance": g0,
            "g0_autorise": g0 <= guards.REJECT_BUDGET,
            "regression_t2": {
                "C": best_c,
                "grille_C": list(C_GRID),
                "exactitude_equilibree_cv_par_C": cv_scores,
                "exactitude_equilibree_cv": cv_balanced,
                "plis": CV_SPLITS,
                "fausses_alertes_figshare_validation": fa_figshare,
                "fausses_alertes_pandit_t1c_hors_pli": fa_pandit,
                "rappel_t2_pandit_dev_hors_pli": t2_recall_oof,
            },
            "images_par_source": counts,
            "choix_figshare": (
                "kaggle_grouped.csv ne porte pas l'identifiant figshare : on prend is_512 == 1 "
                "et classe != notumor (les images 512x512 de tumeurs sont les copies figshare ; "
                "notumor n'existe pas dans figshare)."
            ),
            "donnees_interdites_verifiees": [
                "kaggle test",
                "patients test Pandit (dont 16)",
                "hors sujet",
            ],
        },
        OUT_JSON,
    )
    LOGGER.info("écrit : %s et %s", relative(OUT_WEIGHTS), relative(OUT_JSON))


if __name__ == "__main__":
    sys.exit(main())
