"""Configuration typée et validée du pipeline.

Les hyperparamètres vivent dans ``configs/train.yaml``, pas dans le code. Ils sont
chargés dans des modèles Pydantic : une faute de frappe ou une valeur incorrecte
provoque tout de suite une erreur claire, au lieu de modifier une expérience en
silence.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

# Racine du projet brain-tumor-classifier/ (ce fichier est ml/src/brain_mri/config.py :
# parents[3] remonte de quatre niveaux : brain_mri, src, ml, puis la racine).
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "ml" / "configs" / "train.yaml"


class HeadCropAugConfig(BaseModel):
    """Augmentation du mode de prétraitement ``head_crop`` (entraînement uniquement)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    crop_scale: tuple[float, float] = (
        0.6,
        1.0,
    )  # part de l'image gardée par le recadrage aléatoire
    crop_ratio: tuple[float, float] = (0.8, 1.25)  # rapport largeur/hauteur du recadrage
    zoomout_prob: float = Field(0.3, ge=0, le=1)  # probabilité de dézoom (bandes noires autour)
    zoomout_max: float = Field(1.4, ge=1)  # facteur de dézoom maximal (minimum : 1,0)
    flip_prob: float = Field(0.5, ge=0, le=1)  # probabilité de miroir horizontal
    rotation_degrees: float = Field(10.0, ge=0)  # rotation aléatoire de ± ce nombre de degrés
    contrast: float = Field(0.2, ge=0, le=1)  # variation du contraste (facteur 1 ± contrast)
    gamma_range: tuple[float, float] = (0.8, 1.25)  # plage du gamma tiré au hasard
    blur_prob: float = Field(0.2, ge=0, le=1)  # probabilité de flou gaussien
    blur_sigma: tuple[float, float] = (0.1, 1.5)  # écart-type du flou (pixels)
    noise_prob: float = Field(0.2, ge=0, le=1)  # probabilité de bruit gaussien
    noise_sigma: float = Field(0.03, ge=0)  # écart-type du bruit (échelle 0-1)
    jpeg_prob: float = Field(0.3, ge=0, le=1)  # probabilité de compression JPEG
    jpeg_quality: tuple[int, int] = (50, 95)  # plage de qualité JPEG (1-100)


class DataConfig(BaseModel):
    """Paramètres du pipeline de données (voir ``configs/train.yaml``)."""

    # frozen : l'objet ne peut plus être modifié après sa création.
    # extra="forbid" : une clé inconnue dans le YAML est refusée (détecte les fautes de frappe).
    model_config = ConfigDict(frozen=True, extra="forbid")

    raw_dir: Path = Path("data/raw")  # images téléchargées (Training/ et Testing/)
    processed_dir: Path = Path("data/processed")  # rapports générés (hors git)
    image_size: int = Field(224, gt=0)  # côté de l'image en entrée du ResNet-50
    batch_size: int = Field(32, gt=0)  # nombre d'images par lot
    val_fraction: float = Field(0.15, gt=0, lt=1)  # part de Training réservée à la validation
    seed: int = 42  # graine : rend le découpage et le mélange reproductibles
    num_workers: int = Field(0, ge=0)  # processus de chargement (0 = processus principal)
    # Augmentation légère, appliquée à l'entraînement uniquement.
    flip_prob: float = Field(0.5, ge=0, le=1)  # probabilité de retournement horizontal
    rotation_degrees: float = Field(10.0, ge=0)  # rotation aléatoire de ± ce nombre de degrés
    jitter: float = Field(0.1, ge=0, le=1)  # variation de luminosité et de contraste
    # Mode de prétraitement : "legacy" = comportement historique (étirement direct en 224x224,
    # modèle en ligne) ; "head_crop" = recadrage de la tête, bandes noires, sans étirement.
    preprocessing: Literal["legacy", "head_crop"] = "legacy"
    head_crop_aug: HeadCropAugConfig = HeadCropAugConfig()  # lue seulement en mode head_crop
    # Manifeste de découpage (CSV) : s'il est donné, il remplace le découpage Training/Testing.
    split_manifest: Path | None = None
    manifest_path_column: str = "file"  # colonne : chemin de l'image relatif à raw_dir
    manifest_label_column: str = "class"  # colonne : nom de la classe
    manifest_split_column: str = (
        "part"  # colonne : partie (train, val, test ; autres lignes ignorées)
    )
    # Moyenne et écart-type d'ImageNet : le modèle pré-entraîné les attend.
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)

    @field_validator("raw_dir", "processed_dir")
    @classmethod
    def _resolve_against_project_root(cls, value: Path) -> Path:
        """Transforme un chemin relatif en chemin absolu, à partir de la racine du projet."""
        return value if value.is_absolute() else PROJECT_ROOT / value

    @field_validator("split_manifest")
    @classmethod
    def _resolve_manifest(cls, value: Path | None) -> Path | None:
        """Rend le chemin du manifeste absolu (racine du projet) ; None reste None."""
        if value is None or value.is_absolute():
            return value
        return PROJECT_ROOT / value


class ModelConfig(BaseModel):
    """Paramètres du modèle (voir ``configs/train.yaml``)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    # Architecture : "resnet50" (modèle A) ou "dinov2_s" (modèle B : DINOv2-small gelé + une
    # couche linéaire, voir brain_mri.dinov2_head). Les autres champs ne servent qu'au ResNet.
    architecture: Literal["resnet50", "dinov2_s"] = "resnet50"
    pretrained: bool = True  # partir des poids ImageNet (False seulement dans les tests)
    freeze_backbone: bool = True  # au début, n'entraîner que la nouvelle tête


class PhaseConfig(BaseModel):
    """Réglages d'une phase d'entraînement."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    epochs: int = Field(gt=0)  # nombre maximal d'epochs (l'early stopping peut arrêter avant)
    lr: float = Field(gt=0)  # taux d'apprentissage de l'optimiseur Adam


class TrainConfig(BaseModel):
    """Paramètres de l'entraînement en deux phases (voir ``configs/train.yaml``)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    phase1: PhaseConfig = PhaseConfig(epochs=3, lr=1e-3)  # tête seule
    phase2: PhaseConfig = PhaseConfig(epochs=5, lr=1e-4)  # dégel de layer4 (fine-tuning)
    scheduler_factor: float = Field(0.5, gt=0, lt=1)  # le lr est multiplié par ce facteur...
    scheduler_patience: int = Field(1, ge=0)  # ...après ce nombre d'epochs sans progrès
    early_stopping_patience: int = Field(3, ge=1)  # epochs sans progrès avant d'arrêter la phase
    early_stopping_min_delta: float = Field(1e-4, ge=0)  # amélioration minimale de la val loss
    checkpoint_path: Path = Path("models/best.pt")  # meilleur modèle (hors git : *.pt)
    figures_dir: Path = Path("docs/figures")  # courbes enregistrées en image
    mlflow_experiment: str = "brain-tumor-classifier"  # nom de l'expérience MLflow
    mlflow_dir: Path = Path("mlruns")  # dossier de suivi MLflow (hors git)

    @field_validator("checkpoint_path", "figures_dir", "mlflow_dir")
    @classmethod
    def _resolve_against_project_root(cls, value: Path) -> Path:
        """Transforme un chemin relatif en chemin absolu, à partir de la racine du projet."""
        return value if value.is_absolute() else PROJECT_ROOT / value


class Dinov2HeadConfig(BaseModel):
    """Paramètres de la tête linéaire sur DINOv2 gelé (modèles B0 et B1, section ``dinov2``)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    n_views: int = Field(1, ge=1)  # vues augmentées par image d'entraînement (1 = vue d'évaluation)
    lr: float = Field(1e-3, gt=0)  # taux d'apprentissage d'AdamW
    weight_decay: float = Field(1e-4, ge=0)  # régularisation L2 découplée d'AdamW
    epochs: int = Field(100, gt=0)  # nombre maximal d'epochs (l'arrêt précoce peut arrêter avant)
    patience: int = Field(10, ge=1)  # epochs sans progrès de la loss de validation avant l'arrêt
    min_delta: float = Field(1e-4, ge=0)  # amélioration minimale de la loss de validation
    batch_size: int = Field(256, gt=0)  # caractéristiques par mini-lot d'entraînement de la tête
    extract_batch_size: int = Field(32, gt=0)  # images par lot lors de l'extraction (DINOv2)
    extract_workers: int = Field(0, ge=0)  # processus de chargement pendant l'extraction
    cache_dir: Path = Path("data/cache/dinov2_features")  # caractéristiques en cache (hors git)

    @field_validator("cache_dir")
    @classmethod
    def _resolve_against_project_root(cls, value: Path) -> Path:
        """Transforme un chemin relatif en chemin absolu, à partir de la racine du projet."""
        return value if value.is_absolute() else PROJECT_ROOT / value


class Config(BaseModel):
    """Objet de configuration racine : regroupe les sections du fichier YAML."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    data: DataConfig = DataConfig()
    model: ModelConfig = ModelConfig()
    training: TrainConfig = TrainConfig()
    dinov2: Dinov2HeadConfig = Dinov2HeadConfig()  # lue seulement si architecture = dinov2_s


def load_config(path: Path | str = DEFAULT_CONFIG_PATH) -> Config:
    """Lit et valide un fichier de configuration YAML.

    Args:
        path: Chemin du fichier YAML.

    Returns:
        La configuration validée.

    Raises:
        FileNotFoundError: Si le fichier n'existe pas.
        pydantic.ValidationError: Si une valeur est manquante, inconnue ou invalide.
    """
    with open(path, encoding="utf-8") as file:
        raw = yaml.safe_load(file) or {}  # fichier vide -> dictionnaire vide -> valeurs par défaut
    return Config.model_validate(raw)
