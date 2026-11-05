# =============================================================================
# Image de l'API FastAPI (classifieur d'IRM cérébrales).
#
# Contexte de construction : le dossier brain-tumor-classifier/ (et non api/),
# car l'API dépend du paquet local ../ml.
#   docker build -f api/Dockerfile -t brain-mri-api .
#
# Variables d'environnement :
#   PORT          port d'écoute (8000 en local, 7860 sur Hugging Face Spaces)
#   MODEL_DIR     dossier du modèle (model.safetensors + config.json) ; fixé à /app/model
#   MODEL_REPO_ID dépôt Hugging Face du modèle, utilisé À LA CONSTRUCTION (argument)
#   CORS_ORIGIN   adresse du site autorisée à appeler l'API (ex. http://localhost:8080)
# =============================================================================

# ---- Étape 1 : construction de l'environnement Python ----------------------
FROM python:3.12-slim-trixie AS builder

# uv (gestionnaire de paquets) copié depuis son image officielle, version fixée
COPY --from=ghcr.io/astral-sh/uv:0.13.7 /uv /uvx /bin/

# Pas de Python téléchargé par uv (on utilise celui de l'image) ; compilation des .pyc
ENV UV_PYTHON_DOWNLOADS=0 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv

# Structure /src/ml et /src/api : le chemin relatif ../ml de api/pyproject.toml doit marcher
WORKDIR /src
COPY ml/pyproject.toml ml/uv.lock ml/
COPY ml/src ml/src
COPY api/pyproject.toml api/uv.lock api/
COPY api/app api/app

# Installation figée sur uv.lock, sans les outils de dev, sans installation éditable
# (torch vient de l'index CPU sous Linux, voir pyproject.toml)
WORKDIR /src/api
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-editable

# ---- Étape 2 : image finale (légère, sans uv ni sources de construction) ----
FROM python:3.12-slim-trixie

# Utilisateur non-root (UID 1000, exigé par Hugging Face Spaces)
RUN useradd --create-home --uid 1000 user
WORKDIR /app

# Environnement Python déjà construit
COPY --from=builder --chown=user:user /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

# Code de l'API (sans les tests)
COPY --chown=user:user api/app /app/app

# Modèle : téléchargé une fois à la construction (dépôt public, aucun jeton nécessaire)
ARG MODEL_REPO_ID=david-b59/brain-tumor-classifier
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='${MODEL_REPO_ID}', allow_patterns=['model.safetensors', 'config.json'], local_dir='/app/model')" \
    && chown -R user:user /app/model

USER user
ENV MODEL_DIR=/app/model \
    PORT=8000
EXPOSE 8000

# Santé : la route /health répond 200 quand le modèle est chargé
HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=5 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.environ.get('PORT', '8000') + '/health', timeout=4)"

# Lancement via le shell pour lire la variable PORT
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
