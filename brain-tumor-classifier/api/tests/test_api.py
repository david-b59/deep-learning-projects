"""Tests de l'API (faux modèle à poids aléatoires : on vérifie le contrat, pas la qualité)."""

from __future__ import annotations

import base64
import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.config import MAX_UPLOAD_BYTES, MULTIPART_MARGIN_BYTES, Settings, load_settings
from app.main import create_app
from tests.conftest import CLASSES, image_bytes, make_model_dir


def test_health(client: TestClient) -> None:
    """/health répond « ok » et indique le modèle chargé."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["model_version"].startswith("local:")


def test_predict_returns_the_contract(client: TestClient) -> None:
    """/predict renvoie classe, probabilités (somme 1), PNG Grad-CAM et avertissement."""
    response = client.post("/predict", files={"file": ("irm.png", image_bytes("PNG"), "image/png")})
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"predicted_class", "probabilities", "gradcam_png_base64", "disclaimer"}
    assert body["predicted_class"] in CLASSES
    assert set(body["probabilities"]) == set(CLASSES)
    assert abs(sum(body["probabilities"].values()) - 1.0) < 1e-4
    assert "pas un dispositif médical" in body["disclaimer"]
    Image.open(io.BytesIO(base64.b64decode(body["gradcam_png_base64"]))).verify()  # PNG valide


def test_predict_accepts_jpeg(client: TestClient) -> None:
    """Une image JPEG est acceptée."""
    files = {"file": ("irm.jpg", image_bytes("JPEG"), "image/jpeg")}
    assert client.post("/predict", files=files).status_code == 200


def test_predict_rejects_non_image(client: TestClient) -> None:
    """Un fichier texte est refusé avec une erreur 400."""
    response = client.post("/predict", files={"file": ("a.txt", b"bonjour", "text/plain")})
    assert response.status_code == 400
    assert "image" in response.json()["detail"].lower()


def test_predict_rejects_other_image_format(client: TestClient) -> None:
    """Un format d'image non prévu (GIF) est refusé avec une erreur 400."""
    response = client.post("/predict", files={"file": ("a.gif", image_bytes("GIF"), "image/gif")})
    assert response.status_code == 400


def test_predict_rejects_corrupted_image(client: TestClient) -> None:
    """Un PNG tronqué est refusé avec une erreur 400."""
    truncated = image_bytes("PNG")[:60]
    response = client.post("/predict", files={"file": ("a.png", truncated, "image/png")})
    assert response.status_code == 400


def test_predict_rejects_too_big_file(client: TestClient) -> None:
    """Un fichier de plus de 5 Mo est refusé avec une erreur 413."""
    big = b"0" * (MAX_UPLOAD_BYTES + 1)
    response = client.post("/predict", files={"file": ("a.png", big, "image/png")})
    assert response.status_code == 413


def test_predict_rejects_png_with_broken_data_block(client: TestClient) -> None:
    """Un PNG dont un bloc est cassé (Pillow lève alors SyntaxError) donne un 400, pas un 500."""
    png = bytearray(image_bytes("PNG"))
    start = bytes(png).index(b"IDAT") - 4  # début de la longueur du bloc de pixels
    png[start : start + 4] = (1).to_bytes(4, "big")  # longueur annoncée volontairement fausse
    response = client.post("/predict", files={"file": ("a.png", bytes(png), "image/png")})
    assert response.status_code == 400


def test_predict_rejects_image_with_too_many_pixels(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Une image qui dépasse le nombre de pixels autorisé est refusée avec une erreur 400."""
    monkeypatch.setattr("app.main.MAX_IMAGE_PIXELS", 100)  # l'image de test fait 40 x 50
    response = client.post("/predict", files={"file": ("a.png", image_bytes(), "image/png")})
    assert response.status_code == 400
    assert "pixels" in response.json()["detail"]


def test_oversized_upload_is_rejected_from_the_header(client: TestClient) -> None:
    """Un envoi annoncé trop gros (en-tête Content-Length) est refusé avant d'être reçu."""
    announced = MAX_UPLOAD_BYTES + MULTIPART_MARGIN_BYTES + 1
    response = client.post("/predict", content=b"x", headers={"Content-Length": str(announced)})
    assert response.status_code == 413


def test_oversized_rejection_carries_cors_headers(client: TestClient) -> None:
    """La réponse 413 porte les en-têtes CORS : le navigateur affiche le vrai message d'erreur."""
    announced = MAX_UPLOAD_BYTES + MULTIPART_MARGIN_BYTES + 1
    response = client.post(
        "/predict",
        content=b"x",
        headers={"Content-Length": str(announced), "Origin": "http://front.test"},
    )
    assert response.status_code == 413
    assert response.headers.get("access-control-allow-origin") == "http://front.test"


def test_predict_without_file_is_a_validation_error(client: TestClient) -> None:
    """Sans fichier, FastAPI répond 422."""
    assert client.post("/predict").status_code == 422


def test_predict_internal_error_is_hidden(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Une panne du modèle donne une erreur 500 générique, sans détail technique."""

    def boom(image: Image.Image) -> None:
        """Simule une panne du modèle."""
        raise RuntimeError("secret interne")

    monkeypatch.setattr(client.app.state.model.predictor, "predict", boom)
    response = client.post("/predict", files={"file": ("a.png", image_bytes(), "image/png")})
    assert response.status_code == 500
    assert "secret" not in response.text


def test_metrics_summary(client: TestClient) -> None:
    """/metrics-summary renvoie les scores de test du modèle."""
    body = client.get("/metrics-summary").json()
    assert body["accuracy"] == 0.9
    assert body["class_names"] == CLASSES
    assert body["per_class"]["glioma"]["support"] == 10
    assert len(body["confusion_matrix"]) == 4


def test_metrics_summary_unavailable_without_scores(tmp_path: Path) -> None:
    """Sans scores de test dans config.json, /metrics-summary répond 404."""
    folder = make_model_dir(tmp_path, with_test_scores=False)
    with TestClient(create_app(Settings(folder, "x", "http://front.test"))) as test_client:
        assert test_client.get("/metrics-summary").status_code == 404


def test_cors_allows_only_the_front(client: TestClient) -> None:
    """Seule l'origine du front est autorisée par CORS."""
    allowed = client.get("/health", headers={"Origin": "http://front.test"})
    other = client.get("/health", headers={"Origin": "http://mechant.test"})
    assert allowed.headers.get("access-control-allow-origin") == "http://front.test"
    assert "access-control-allow-origin" not in other.headers


def test_swagger_documentation_is_served(client: TestClient) -> None:
    """La documentation Swagger et le schéma OpenAPI sont disponibles."""
    assert client.get("/docs").status_code == 200
    paths = client.get("/openapi.json").json()["paths"]
    assert {"/health", "/predict", "/metrics-summary"} <= set(paths)


def test_load_settings_reads_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Les réglages viennent des variables d'environnement."""
    monkeypatch.setenv("MODEL_DIR", str(tmp_path))
    monkeypatch.setenv("MODEL_REPO_ID", "alice/modele")
    monkeypatch.setenv("CORS_ORIGIN", "http://exemple.test")
    settings = load_settings()
    assert settings.model_dir == tmp_path
    assert settings.model_repo_id == "alice/modele"
    assert settings.cors_origin == "http://exemple.test"
