"""
Suíte de Testes Automatizados para a API REST MNIST.
Testa conformidade dos endpoints /health e /predict com TestClient e Lifespan ativo.
"""

import json
import pytest
from fastapi.testclient import TestClient
from main import app


@pytest.fixture(scope="module")
def client():
    """
    Fixture que gerencia o ciclo de vida (Lifespan) durante os testes.
    Dispara o startup (carregamento do modelo Keras em RAM) e o shutdown gracioso.
    """
    with TestClient(app) as test_client:
        yield test_client


def test_health_check_retorna_200_e_modelo_carregado(client):
    """Valida se o endpoint /health responde 200 OK e acusa modelo em RAM."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["model_version"] == "MLP-Keras-v1.0"
    assert "timestamp" in data


def test_predict_dado_valido_retorna_200_e_predicao(client):
    """Valida inferência com vetor canônico do dígito 7."""
    with open("data/exemplo_digito_7.json", "r") as f:
        payload = json.load(f)

    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["predicted_class"] == 7
    assert data["confidence"] > 0.90
    assert len(data["probabilities"]) == 10
    assert data["inference_time_ms"] >= 0.0


def test_predict_invalido_retorna_422_unprocessable_entity(client):
    """Valida se dados com dimensão incorreta são rejeitados pelo Pydantic com HTTP 422."""
    payload_invalido = {"features": [0.5] * 10}  # Apenas 10 pixels em vez de 784
    response = client.post("/predict", json=payload_invalido)
    assert response.status_code == 422
    data = response.json()
    assert "detail" in data
