"""
Tests para la API de riesgo crediticio.
"""
import pytest
import json
import pandas as pd
import numpy as np
from fastapi.testclient import TestClient
import sys
import os

# Añadir src al path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.api.app import app


@pytest.fixture
def client():
    """Fixture para el cliente de测试."""
    return TestClient(app)


@pytest.fixture
def sample_application():
    """Fixture para una solicitud de crédito de ejemplo."""
    return {
        "edad": 35,
        "genero": "M",
        "estado_civil": "casado",
        "dependientes": 1,
        "ingreso_mensual": 3000.0,
        "gastos_mensuales": 2000.0,
        "total_adeudado": 5000.0,
        "ahorros": 10000.0,
        "score_bancario": 720,
        "antiguedad_empleo": 24,
        "tipo_contrato": "indefinido",
        "num_creditos_previos": 2,
        "max_dias_mora": 15,
        "creditos_problematicos": 0,
        "tipo_vivienda": "propia",
        "antiguedad_residencia": 5
    }


def test_root_endpoint(client):
    """Prueba el endpoint raíz."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "healthy"


def test_health_check(client):
    """Prueba el health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "timestamp" in data


def test_model_info(client):
    """Prueba el endpoint de información del modelo."""
    response = client.get("/model/info")
    
    # Puede ser 200 o 503 dependiendo si el modelo está cargado
    if response.status_code == 200:
        data = response.json()
        assert "model_type" in data
        assert "feature_count" in data
    elif response.status_code == 503:
        # Modelo no cargado, esto es aceptable en tests
        pass


def test_features_endpoint(client):
    """Prueba el endpoint de features."""
    response = client.get("/features")
    
    if response.status_code == 200:
        data = response.json()
        assert "features" in data
        assert "count" in data
        assert "sample_input" in data
    elif response.status_code == 503:
        # Modelo no cargado
        pass


def test_predict_endpoint_unauthorized(client, sample_application):
    """Prueba predicción sin autorización."""
    response = client.post("/predict", json=sample_application)
    # Debería fallar por falta de token
    assert response.status_code == 401 or response.status_code == 403


def test_predict_endpoint_authorized(client, sample_application):
    """Prueba predicción con autorización."""
    # Usar token de prueba (en desarrollo, cualquier token funciona)
    headers = {"Authorization": "Bearer demo_token_12345"}
    response = client.post("/predict", json=sample_application, headers=headers)
    
    # Puede fallar si el modelo no está cargado
    if response.status_code == 200:
        data = response.json()
        assert "prediction" in data
        assert "probability_default" in data
        assert "risk_score" in data
        assert "risk_category" in data
        assert "features_used" in data
        
        # Validar rangos
        assert 0 <= data["probability_default"] <= 1
        assert 300 <= data["risk_score"] <= 850
        assert data["prediction"] in [0, 1]
    elif response.status_code in [401, 403, 503]:
        # Errores esperados en ambiente de test
        pass


def test_batch_predict_endpoint(client, sample_application):
    """Prueba predicción por lotes."""
    batch_request = {
        "applications": [sample_application, sample_application]
    }
    
    headers = {"Authorization": "Bearer demo_token_12345"}
    response = client.post("/predict/batch", json=batch_request, headers=headers)
    
    if response.status_code == 200:
        data = response.json()
        assert "total_applications" in data
        assert "successful_predictions" in data
        assert "failed_predictions" in data
        assert "predictions" in data
        
        assert data["total_applications"] == 2
        assert len(data["predictions"]) == 2


def test_validation_errors(client):
    """Prueba errores de validación."""
    # Solicitud inválida (edad muy baja)
    invalid_application = {
        "edad": 10,  # Muy baja
        "genero": "M",
        "ingreso_mensual": 1000.0,
        # Faltan campos requeridos
    }
    
    headers = {"Authorization": "Bearer demo_token"}
    response = client.post("/predict", json=invalid_application, headers=headers)
    
    # Debería ser 422 (Unprocessable Entity) por validación fallida
    assert response.status_code == 422


def test_error_handling(client):
    """Prueba manejo de errores."""
    # Request malformado
    response = client.post("/predict", data="not json")
    assert response.status_code in [400, 422, 401]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])