"""Formats des réponses de l'API (validés par Pydantic, affichés dans Swagger)."""

from __future__ import annotations

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Réponse de ``GET /health``."""

    status: str = Field(description="« ok » si l'API fonctionne et que le modèle est chargé")
    model_version: str = Field(description="Origine du modèle chargé (dossier local ou dépôt)")


class PredictionResponse(BaseModel):
    """Réponse de ``POST /predict``."""

    predicted_class: str = Field(description="Classe la plus probable")
    probabilities: dict[str, float] = Field(description="Probabilité de chaque classe (somme = 1)")
    gradcam_png_base64: str = Field(
        description="IRM avec la carte Grad-CAM superposée (PNG base64)"
    )
    disclaimer: str = Field(
        description="Rappel : outil de démonstration, pas un dispositif médical"
    )


class ClassScores(BaseModel):
    """Scores d'une classe sur le jeu de test."""

    precision: float
    recall: float
    f1: float
    support: int = Field(description="Nombre d'images de cette classe dans le jeu de test")


class MetricsSummary(BaseModel):
    """Réponse de ``GET /metrics-summary`` : scores du modèle sur le jeu de test."""

    accuracy: float = Field(description="Part d'images du test correctement classées")
    class_names: list[str] = Field(description="Noms des classes, dans l'ordre de la matrice")
    per_class: dict[str, ClassScores]
    confusion_matrix: list[list[int]] = Field(
        description="Lignes : vraie classe. Colonnes : classe prédite."
    )
