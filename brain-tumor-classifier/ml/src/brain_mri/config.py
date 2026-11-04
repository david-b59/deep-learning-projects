"""Configuration typée et validée du pipeline.

Les hyperparamètres vivent dans ``configs/train.yaml``, pas dans le code. Ils sont
chargés dans des modèles Pydantic : une faute de frappe ou une valeur incorrecte
provoque tout de suite une erreur claire, au lieu de modifier une expérience en
silence.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

# Racine du projet brain-tumor-classifier/ (ce fichier est ml/src/brain_mri/config.py :
# parents[3] remonte de quatre niveaux : brain_mri, src, ml, puis la racine).
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "ml" / "configs" / "train.yaml"


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
    # Moyenne et écart-type d'ImageNet : le modèle pré-entraîné les attend.
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)

    @field_validator("raw_dir", "processed_dir")
    @classmethod
    def _resolve_against_project_root(cls, value: Path) -> Path:
        """Transforme un chemin relatif en chemin absolu, à partir de la racine du projet."""
        return value if value.is_absolute() else PROJECT_ROOT / value


class ModelConfig(BaseModel):
    """Paramètres du modèle (voir ``configs/train.yaml``)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

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


class Config(BaseModel):
    """Objet de configuration racine : regroupe les sections du fichier YAML."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    data: DataConfig = DataConfig()
    model: ModelConfig = ModelConfig()
    training: TrainConfig = TrainConfig()


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
