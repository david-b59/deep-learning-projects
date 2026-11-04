"""Application FastAPI : sert le classifieur d'IRM cérébrales.

Routes :
    - ``GET /health`` : l'API est-elle vivante, quel modèle est chargé ;
    - ``GET /metrics-summary`` : scores du modèle sur le jeu de test ;
    - ``POST /predict`` : envoie une IRM (JPG ou PNG, 5 Mo maximum), reçoit la classe, les
      probabilités et l'image Grad-CAM.

Swagger (documentation interactive générée automatiquement) : ``/docs``.

Lancer en local (depuis ``brain-tumor-classifier/api/``) ::

    uv run uvicorn app.main:app --port 8000
"""

from __future__ import annotations

import io
import logging
import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError

from app.config import (
    MAX_IMAGE_PIXELS,
    MAX_UPLOAD_BYTES,
    MULTIPART_MARGIN_BYTES,
    Settings,
    load_settings,
)
from app.model_loader import LoadedModel, load_model
from app.schemas import HealthResponse, MetricsSummary, PredictionResponse

LOGGER = logging.getLogger(__name__)
ALLOWED_FORMATS = {"JPEG", "PNG"}  # formats d'image acceptés


def read_image(data: bytes) -> Image.Image:
    """Ouvre et vérifie une image reçue.

    Args:
        data: Contenu brut du fichier envoyé.

    Returns:
        L'image PIL, entièrement décodée.

    Raises:
        HTTPException: 400 si le fichier n'est pas une image JPG/PNG valide, ou trop grande.
    """
    try:
        image = Image.open(io.BytesIO(data))
        if image.format not in ALLOWED_FORMATS:
            raise HTTPException(400, "Format non accepté : envoie une image JPG ou PNG.")
        if image.width * image.height > MAX_IMAGE_PIXELS:
            raise HTTPException(400, "Image trop grande en nombre de pixels.")
        image.load()  # décode vraiment les pixels : détecte les fichiers corrompus
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as error:
        # SyntaxError : Pillow la lève pour certains PNG corrompus ; la bombe de décompression
        # (DecompressionBombError) est une ValueError.
        raise HTTPException(400, "Fichier illisible : ce n'est pas une image valide.") from error
    return image


def create_app(settings: Settings | None = None) -> FastAPI:
    """Fabrique l'application (les tests lui passent leurs propres réglages).

    Args:
        settings: Réglages ; par défaut, lus dans l'environnement.

    Returns:
        L'application FastAPI. Le modèle est chargé au démarrage, une seule fois.
    """
    settings = settings or load_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        """Charge le modèle au démarrage et le range dans ``app.state``."""
        app.state.model = load_model(settings)
        # Grad-CAM pose des « hooks » sur le modèle partagé : une seule prédiction à la fois.
        app.state.lock = threading.Lock()
        yield

    app = FastAPI(
        title="Brain Tumor Classifier",
        description="Classifie une IRM cérébrale (4 classes) et explique la décision "
        "avec Grad-CAM. Démonstration technique, pas un dispositif médical.",
        version="0.1.0",
        lifespan=lifespan,
    )

    # Ordre des middlewares : le dernier ajouté est le plus extérieur. CORS doit entourer le
    # contrôle de taille, pour que la réponse 413 porte aussi les en-têtes CORS.
    @app.middleware("http")
    async def reject_oversized_upload(request: Request, call_next):  # noqa: ANN001, ANN202
        """Refuse tout de suite un envoi annoncé trop gros, avant que le corps soit reçu.

        Starlette écrit tout le corps d'une requête multipart sur disque avant que la route
        s'exécute : sans ce contrôle sur l'en-tête ``Content-Length``, un envoi énorme serait
        d'abord reçu en entier. Le contrôle fait dans la route reste le filet de sécurité
        (en-tête absent ou mensonger).
        """
        length = request.headers.get("content-length", "")
        too_big = length.isdigit() and int(length) > MAX_UPLOAD_BYTES + MULTIPART_MARGIN_BYTES
        if request.url.path == "/predict" and too_big:
            return JSONResponse(
                {"detail": "Fichier trop volumineux (5 Mo maximum)."}, status_code=413
            )
        return await call_next(request)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[settings.cors_origin],  # seulement le front, pas tout Internet
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    def model_of(request: Request) -> LoadedModel:
        """Récupère le modèle chargé au démarrage."""
        return request.app.state.model

    @app.get("/health", response_model=HealthResponse, summary="État de l'API")
    def health(request: Request) -> HealthResponse:
        """Indique que l'API fonctionne et quel modèle est chargé."""
        return HealthResponse(status="ok", model_version=model_of(request).version)

    @app.get("/metrics-summary", response_model=MetricsSummary, summary="Scores sur le test")
    def metrics_summary(request: Request) -> MetricsSummary:
        """Renvoie les scores du modèle sur le jeu de test (pour la page Résultats)."""
        description = model_of(request).description
        if "test" not in description:
            raise HTTPException(404, "Les scores de test ne sont pas disponibles.")
        test = description["test"]
        return MetricsSummary(
            accuracy=test["accuracy"],
            class_names=description["class_names"],
            per_class=test["per_class"],
            confusion_matrix=test["confusion_matrix"],
        )

    @app.post("/predict", response_model=PredictionResponse, summary="Classer une IRM")
    def predict(request: Request, file: UploadFile = File(...)) -> PredictionResponse:  # noqa: B008
        """Classe une IRM (JPG ou PNG, 5 Mo maximum) et renvoie la carte Grad-CAM."""
        data = file.file.read(MAX_UPLOAD_BYTES + 1)  # un octet de plus : détecte le dépassement
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "Fichier trop volumineux (5 Mo maximum).")
        image = read_image(data)

        loaded = model_of(request)
        try:
            with request.app.state.lock:
                result = loaded.predictor.predict(image)
        except Exception as error:  # erreur inattendue : on la journalise, sans la montrer
            LOGGER.exception("Échec de la prédiction")
            raise HTTPException(500, "Erreur interne lors de la prédiction.") from error
        return PredictionResponse(
            predicted_class=result.predicted_class,
            probabilities=result.probabilities,
            gradcam_png_base64=result.gradcam_png_base64,
            disclaimer=result.disclaimer,
        )

    return app


app = create_app()  # l'application utilisée par uvicorn : ``app.main:app``
