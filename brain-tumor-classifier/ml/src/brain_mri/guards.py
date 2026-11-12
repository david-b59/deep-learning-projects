"""Garde-fous du modèle A1 (étape 11c-1) : couleur, température, distance, confiance, détecteur T2.

Les règles sont écrites AVANT toute mesure dans ``ml/splits/guards_rules.md`` (elles font loi).
Ce module contient la logique pure (numpy / torch, aucun fichier d'images) :

    - :func:`color_statistic` : part de pixels colorés, sur l'image NATIVE (avant recadrage).
      Volontairement simple : elle sera recopiée en TypeScript (navigateur) à l'étape 11c-4.
    - :class:`GuardHead` : un ``nn.Module`` exportable en ONNX. Entrées : logits d'A1 et 2048
      caractéristiques (avant-dernière couche). Sorties : probabilités après température,
      distance de Mahalanobis minimale aux 4 centres de classe, probabilité « T2 ».
    - :func:`verdict` : applique l'ordre ``color`` -> ``t2_like`` -> ``out_of_distribution`` /
      ``low_confidence`` -> ``accepted``.
    - Les fonctions d'ajustement (:func:`fit_temperature`, :func:`fit_mahalanobis`,
      :func:`choose_tau_p`, :func:`choose_tau_t2`...) et l'assertion d'interdiction
      :func:`assert_no_forbidden_data`.
    - La sauvegarde / lecture en safetensors (:func:`save_guards`, :func:`load_guards`).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from torch import nn

COLOR_MIN_MAX_CHANNEL = 16  # un pixel compte seulement si son canal maximal dépasse cette valeur
COLOR_SPREAD = 24  # un pixel est « coloré » si (max - min entre canaux) dépasse cette valeur
COLOR_FLOOR = 0.01  # plancher du seuil couleur (1 %)
COLOR_FACTOR = 2.0  # seuil = 2 x le maximum observé sur Kaggle train + val
PCA_COMPONENTS = 256  # composantes de l'ACP avant Mahalanobis
TAU_D_PERCENTILE = 97.0  # tau_D = ce percentile des distances de la validation
REJECT_BUDGET = 0.05  # part maximale de la validation refusée (distance OU confiance)
T2_MAX_FALSE_ALARM_FIGSHARE = 0.02  # fausses alertes T2 tolérées sur les images figshare (val)
T2_MAX_FALSE_ALARM_PANDIT = 0.05  # fausses alertes T2 tolérées sur les T1c Pandit dev (hors pli)

VERDICTS = ("color", "t2_like", "out_of_distribution", "low_confidence", "accepted")


# --------------------------------------------------------------------------- couleur


def color_statistic(image: np.ndarray) -> float:
    """Calcule la part de pixels colorés d'une image native.

    On ne regarde que les pixels dont le canal maximal dépasse 16 (on ignore le fond noir).
    La statistique est la part de ces pixels dont l'écart (max - min) entre canaux dépasse 24.
    Une image en niveaux de gris (2 dimensions, ou 3 canaux identiques) vaut 0.

    Args:
        image: Tableau ``(H, W)`` ou ``(H, W, C)`` d'entiers 0-255 (C = 3 ou 4 ; l'éventuel
            canal alpha est ignoré).

    Returns:
        Un nombre entre 0 et 1 (0 s'il n'y a aucun pixel au-dessus de 16).
    """
    if image.ndim == 2:
        return 0.0  # un seul canal : aucun écart de couleur possible
    pixels = image[..., :3].reshape(-1, 3).astype(np.int16)  # int16 : évite le dépassement uint8
    highest = pixels.max(axis=1)
    spread = highest - pixels.min(axis=1)
    kept = highest > COLOR_MIN_MAX_CHANNEL
    if not kept.any():
        return 0.0
    return float((spread[kept] > COLOR_SPREAD).mean())


def color_threshold(observed_max: float) -> float:
    """Seuil couleur : 2 fois le maximum observé sur Kaggle train + val, avec un plancher de 1 %.

    Args:
        observed_max: Plus grande statistique couleur vue sur Kaggle train + val.

    Returns:
        Le seuil au-dessus duquel une image est refusée comme « couleur ».
    """
    return max(COLOR_FLOOR, COLOR_FACTOR * observed_max)


# --------------------------------------------------------------------------- module exportable


class GuardHead(nn.Module):
    """Tête des garde-fous : température, distance de Mahalanobis et détecteur T2.

    Tous les paramètres sont des ``buffers`` (aucun n'est entraîné par descente de gradient) :
    ils sont sauvegardés avec le module et exportables en ONNX. Les seuils sont aussi stockés
    (``thresholds`` = [couleur, tau_D, tau_P, tau_T2]).

    Distance : ``z = (x - moyenne) @ composantes.T`` (ACP), puis pour chaque centre ``c`` :
    ``d2 = (z - c) @ précision @ (z - c).T`` ; la sortie est ``sqrt(min_c d2)``.
    """

    def __init__(
        self,
        n_features: int = 2048,
        n_components: int = PCA_COMPONENTS,
        n_classes: int = 4,
    ) -> None:
        """Crée la tête avec des valeurs neutres (à remplacer par l'ajustement ou le chargement).

        Args:
            n_features: Nombre de caractéristiques d'entrée (2048 pour le ResNet-50).
            n_components: Composantes de l'ACP.
            n_classes: Nombre de classes (centres).
        """
        super().__init__()
        self.register_buffer("temperature", torch.ones(()))  # logits / température
        self.register_buffer("pca_mean", torch.zeros(n_features))
        self.register_buffer("pca_components", torch.zeros(n_components, n_features))
        self.register_buffer("precision", torch.eye(n_components))  # inverse de la covariance
        self.register_buffer("centers", torch.zeros(n_classes, n_components))
        self.register_buffer("t2_weight", torch.zeros(n_features))  # régression logistique
        self.register_buffer("t2_bias", torch.zeros(()))
        self.register_buffer("thresholds", torch.zeros(4))  # couleur, tau_D, tau_P, tau_T2

    def forward(
        self, logits: torch.Tensor, features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Calcule les trois sorties des garde-fous.

        Args:
            logits: Logits d'A1, ``(N, classes)``.
            features: Caractéristiques avant la dernière couche, ``(N, 2048)``.

        Returns:
            ``(probabilités après température (N, classes), distance de Mahalanobis minimale
            (N,), probabilité T2 (N,))``.
        """
        probabilities = torch.softmax(logits / self.temperature, dim=1)
        projected = (features - self.pca_mean) @ self.pca_components.T  # (N, composantes)
        diff = projected[:, None, :] - self.centers[None, :, :]  # (N, classes, composantes)
        squared = ((diff @ self.precision) * diff).sum(dim=2)  # (N, classes)
        distance = squared.min(dim=1).values.clamp_min(0.0).sqrt()
        t2_probability = torch.sigmoid(features @ self.t2_weight + self.t2_bias)
        return probabilities, distance, t2_probability

    def threshold_values(self) -> GuardThresholds:
        """Relit les seuils stockés sous forme lisible.

        Returns:
            Les quatre seuils.
        """
        color, tau_d, tau_p, tau_t2 = (float(v) for v in self.thresholds)
        return GuardThresholds(color=color, tau_d=tau_d, tau_p=tau_p, tau_t2=tau_t2)


@dataclass(frozen=True)
class GuardThresholds:
    """Les quatre seuils des garde-fous."""

    color: float  # part de pixels colorés au-dessus de laquelle l'image est refusée
    tau_d: float  # distance de Mahalanobis au-dessus de laquelle l'image est hors distribution
    tau_p: float  # confiance (après température) en dessous de laquelle l'image est incertaine
    tau_t2: float  # probabilité T2 au-dessus de laquelle l'image est signalée « T2 »


def verdict(
    color: float, t2_prob: float, distance: float, max_prob: float, thresholds: GuardThresholds
) -> str:
    """Applique l'ordre des garde-fous de ``guards_rules.md``.

    Ordre : couleur (refusée) -> T2 (signalée) -> hors distribution ou confiance basse
    (incertaine) -> acceptée.

    Args:
        color: Statistique couleur de l'image native.
        t2_prob: Probabilité T2 du détecteur.
        distance: Distance de Mahalanobis minimale.
        max_prob: Plus grande probabilité de classe, après température.
        thresholds: Les seuils.

    Returns:
        Un élément de :data:`VERDICTS`.
    """
    if color > thresholds.color:
        return "color"
    if t2_prob > thresholds.tau_t2:
        return "t2_like"
    if distance > thresholds.tau_d:
        return "out_of_distribution"
    if max_prob < thresholds.tau_p:
        return "low_confidence"
    return "accepted"


# --------------------------------------------------------------------------- ajustement


def expected_calibration_error(
    probabilities: np.ndarray, labels: np.ndarray, bins: int = 15
) -> float:
    """Calcule l'ECE (erreur de calibration attendue) avec des intervalles de confiance égaux.

    Args:
        probabilities: Probabilités par classe, ``(N, classes)``.
        labels: Classes vraies, ``(N,)``.
        bins: Nombre d'intervalles de confiance.

    Returns:
        La moyenne pondérée de |exactitude - confiance| sur les intervalles.
    """
    confidence = probabilities.max(axis=1)
    correct = (probabilities.argmax(axis=1) == labels).astype(np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for low, high in zip(edges[:-1], edges[1:], strict=True):
        in_bin = (confidence > low) & (confidence <= high)
        if in_bin.any():
            ece += in_bin.mean() * abs(correct[in_bin].mean() - confidence[in_bin].mean())
    return float(ece)


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """Ajuste la température par LBFGS sur la NLL (log-vraisemblance négative).

    On optimise ``log T`` (donc T > 0). Diviser les logits par un scalaire positif ne change
    jamais l'ordre des classes : l'argmax reste identique.

    Args:
        logits: Logits de validation ``(N, classes)``.
        labels: Classes vraies ``(N,)``.

    Returns:
        La température T.
    """
    x = torch.as_tensor(logits, dtype=torch.float64)
    y = torch.as_tensor(labels, dtype=torch.long)
    log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200, line_search_fn="strong_wolfe")

    def closure() -> torch.Tensor:
        """Calcule la NLL pour la température courante et ses gradients."""
        optimizer.zero_grad()
        loss = nn.functional.cross_entropy(x / log_t.exp(), y)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_t.exp().item())


@dataclass
class MahalanobisFit:
    """Paramètres ajustés de la distance de Mahalanobis (tout en float32, prêt pour GuardHead)."""

    mean: np.ndarray  # moyenne des caractéristiques du train, (2048,)
    components: np.ndarray  # composantes de l'ACP, (composantes, 2048)
    precision: np.ndarray  # inverse de la covariance commune LedoitWolf, (composantes, composantes)
    centers: np.ndarray  # centres de classe dans l'espace ACP, (classes, composantes)


def fit_mahalanobis(
    features: np.ndarray, labels: np.ndarray, n_components: int = PCA_COMPONENTS
) -> MahalanobisFit:
    """Ajuste ACP + covariance commune LedoitWolf + centres de classe sur le train.

    La covariance commune est estimée par LedoitWolf sur les résidus (point moins le centre
    de sa classe), donc une seule covariance pour les 4 classes.

    Args:
        features: Caractéristiques du train, ``(N, 2048)``.
        labels: Classes du train ``(N,)``.
        n_components: Nombre de composantes de l'ACP.

    Returns:
        Les paramètres ajustés.
    """
    x = features.astype(np.float64)
    pca = PCA(n_components=n_components, svd_solver="full").fit(x)
    projected = pca.transform(x)
    classes = np.unique(labels)
    centers = np.stack([projected[labels == c].mean(axis=0) for c in classes])
    residuals = projected - centers[np.searchsorted(classes, labels)]
    covariance = LedoitWolf(assume_centered=True).fit(residuals)  # résidus déjà centrés
    return MahalanobisFit(
        mean=pca.mean_.astype(np.float32),
        components=pca.components_.astype(np.float32),
        precision=covariance.precision_.astype(np.float32),
        centers=centers.astype(np.float32),
    )


def choose_tau_d(val_distances: np.ndarray) -> float:
    """Calcule tau_D : le 97e percentile des distances de la validation.

    Args:
        val_distances: Distances de Mahalanobis de la validation.

    Returns:
        Le seuil de distance.
    """
    return float(np.percentile(val_distances, TAU_D_PERCENTILE))


def choose_tau_p(val_distances: np.ndarray, val_max_prob: np.ndarray, tau_d: float) -> float:
    """Calcule tau_P : la plus grande valeur gardant la part refusée de la validation <= 5 %.

    Une image est refusée si sa distance dépasse tau_D OU si sa confiance est < tau_P.

    Args:
        val_distances: Distances de la validation.
        val_max_prob: Plus grande probabilité (après température) de la validation.
        tau_d: Seuil de distance déjà choisi.

    Returns:
        tau_P (0 si même sans règle de confiance la distance refuse déjà plus de 5 %).
    """
    far = val_distances > tau_d
    best = 0.0
    for candidate in np.unique(val_max_prob):  # croissant : la part refusée ne fait que monter
        rejected = (far | (val_max_prob < candidate)).mean()
        if rejected <= REJECT_BUDGET:
            best = float(candidate)
        else:
            break
    return best


def rejected_fraction(
    distances: np.ndarray, max_prob: np.ndarray, tau_d: float, tau_p: float
) -> float:
    """Part d'images refusées par la distance OU la confiance (G0 sur la validation).

    Args:
        distances: Distances de Mahalanobis.
        max_prob: Plus grande probabilité de classe.
        tau_d: Seuil de distance.
        tau_p: Seuil de confiance.

    Returns:
        La part refusée, entre 0 et 1.
    """
    return float(((distances > tau_d) | (max_prob < tau_p)).mean())


def choose_tau_t2(figshare_val_prob: np.ndarray, pandit_t1_oof_prob: np.ndarray) -> float:
    """Calcule tau_T2 : le plus petit seuil respectant les deux taux de fausses alertes.

    Une image est signalée T2 si sa probabilité T2 dépasse le seuil. Conditions : au plus 2 %
    de fausses alertes sur les images figshare de la validation Kaggle ET au plus 5 % sur les
    coupes T1c de Pandit développement (prédictions hors pli).

    Args:
        figshare_val_prob: Probabilités T2 des images figshare de la validation (T1c certain).
        pandit_t1_oof_prob: Probabilités T2 hors pli des coupes T1c de Pandit développement.

    Returns:
        Le seuil.
    """
    candidates = np.unique(np.concatenate([figshare_val_prob, pandit_t1_oof_prob, [1.0]]))
    for candidate in candidates:  # croissant : le premier qui convient est le plus petit
        if (figshare_val_prob > candidate).mean() <= T2_MAX_FALSE_ALARM_FIGSHARE and (
            pandit_t1_oof_prob > candidate
        ).mean() <= T2_MAX_FALSE_ALARM_PANDIT:
            return float(candidate)
    return 1.0  # inatteignable : le candidat 1.0 ne signale jamais rien


def fold_standardization(
    weight: np.ndarray, bias: float, mean: np.ndarray, scale: np.ndarray
) -> tuple[np.ndarray, float]:
    """Replie une standardisation (x - moyenne) / écart-type dans les poids d'une régression.

    ``w.((x - m) / s) + b = (w / s).x + (b - sum(w * m / s))`` : le module n'a ainsi besoin
    d'aucune étape de standardisation.

    Args:
        weight: Poids appris sur les caractéristiques standardisées, ``(2048,)``.
        bias: Biais appris.
        mean: Moyenne utilisée pour standardiser.
        scale: Écart-type utilisé pour standardiser.

    Returns:
        ``(poids sur les caractéristiques brutes, biais)``.
    """
    raw_weight = weight / scale
    return raw_weight, float(bias - (raw_weight * mean).sum())


# --------------------------------------------------------------------------- interdictions


def assert_no_forbidden_data(
    kaggle_rows: list[dict],
    pandit_dev_patients: set[str],
    forbidden_patients: set[str],
    used_kaggle_parts: set[str],
    used_offtopic: set[str],
) -> None:
    """Vérifie qu'aucune donnée interdite n'entre dans l'ajustement (règle de guards_rules.md).

    Interdits : la partie ``test`` de Kaggle, les patients du test Pandit (30 tirés à l'étape
    11b-7, patient 16 compris) et tout le jeu hors sujet.

    Args:
        kaggle_rows: Lignes de Kaggle utilisées pour ajuster (clé ``part`` lue).
        pandit_dev_patients: Patients Pandit utilisés pour ajuster.
        forbidden_patients: Patients Pandit interdits (test 11b-7 + patient 16).
        used_kaggle_parts: Parties Kaggle réellement utilisées (doit être incluse dans train, val).
        used_offtopic: Fichiers hors sujet utilisés (doit être vide).

    Raises:
        AssertionError: Si une donnée interdite est présente.
    """
    parts = {row["part"] for row in kaggle_rows} | set(used_kaggle_parts)
    assert parts <= {"train", "val"}, f"Parties Kaggle interdites dans l'ajustement : {parts}"
    leaked = set(pandit_dev_patients) & set(forbidden_patients)
    assert not leaked, f"Patients Pandit du test utilisés dans l'ajustement : {sorted(leaked)}"
    assert not set(used_offtopic), "Le jeu hors sujet ne doit servir à aucun ajustement"


# --------------------------------------------------------------------------- sauvegarde


def save_guards(head: GuardHead, path: Path) -> None:
    """Enregistre les paramètres de la tête en safetensors (tenseurs seulement).

    Args:
        head: La tête ajustée.
        path: Fichier ``.safetensors`` à écrire (le dossier est créé si besoin).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file({k: v.contiguous() for k, v in head.state_dict().items()}, str(path))


def load_guards(path: Path) -> GuardHead:
    """Relit une tête enregistrée par :func:`save_guards` (les dimensions viennent du fichier).

    Args:
        path: Fichier ``.safetensors``.

    Returns:
        La tête, en mode évaluation.
    """
    tensors = load_file(str(path))
    n_components, n_features = tensors["pca_components"].shape
    head = GuardHead(n_features, n_components, tensors["centers"].shape[0])
    head.load_state_dict(tensors)
    return head.eval()


def write_json(data: dict, path: Path) -> None:
    """Écrit un dictionnaire en JSON lisible (UTF-8, indenté).

    Args:
        data: Données à écrire.
        path: Fichier de sortie (le dossier est créé si besoin).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def thresholds_dict(thresholds: GuardThresholds) -> dict[str, float]:
    """Convertit les seuils en dictionnaire pour le JSON.

    Args:
        thresholds: Les seuils.

    Les valeurs sont arrondies en float32, comme dans les buffers du module : le JSON, l'ONNX et
    le navigateur lisent ainsi exactement les mêmes seuils (parité G6).

    Returns:
        Un dictionnaire nom -> valeur.
    """
    return {name: float(np.float32(value)) for name, value in asdict(thresholds).items()}
