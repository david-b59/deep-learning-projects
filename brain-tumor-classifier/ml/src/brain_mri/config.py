"""Typed, validated configuration for the pipeline.

Hyperparameters live in ``configs/train.yaml``, not in the code. They are loaded
into Pydantic models, so a typo or a wrong value fails immediately with a clear
message instead of silently changing an experiment.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

# brain-tumor-classifier/ (this file is ml/src/brain_mri/config.py)
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "ml" / "configs" / "train.yaml"


class DataConfig(BaseModel):
    """Settings of the data pipeline (see ``configs/train.yaml``)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    raw_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")
    image_size: int = Field(224, gt=0)
    batch_size: int = Field(32, gt=0)
    val_fraction: float = Field(0.15, gt=0, lt=1)
    seed: int = 42
    num_workers: int = Field(0, ge=0)
    # Light augmentation, applied to the training set only.
    flip_prob: float = Field(0.5, ge=0, le=1)
    rotation_degrees: float = Field(10.0, ge=0)
    jitter: float = Field(0.1, ge=0, le=1)  # brightness and contrast
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)

    @field_validator("raw_dir", "processed_dir")
    @classmethod
    def _resolve_against_project_root(cls, value: Path) -> Path:
        """Turn relative paths into absolute ones, relative to the project root."""
        return value if value.is_absolute() else PROJECT_ROOT / value


class Config(BaseModel):
    """Root configuration object."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    data: DataConfig = DataConfig()


def load_config(path: Path | str = DEFAULT_CONFIG_PATH) -> Config:
    """Read and validate a YAML configuration file.

    Args:
        path: Path of the YAML file.

    Returns:
        The validated configuration.

    Raises:
        FileNotFoundError: If the file does not exist.
        pydantic.ValidationError: If a value is missing, unknown or invalid.
    """
    with open(path, encoding="utf-8") as file:
        raw = yaml.safe_load(file) or {}
    return Config.model_validate(raw)
