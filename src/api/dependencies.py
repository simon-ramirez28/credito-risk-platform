"""
Dependencias y utilidades para la API.
"""

import joblib
import json
from typing import Optional
from fastapi import HTTPException, status
from .config import settings


def get_model():
    """
    Dependency para obtener el modelo cargado.
    En una implementación real, esto podría manejar caché y reload.
    """
    try:
        model = joblib.load(settings.model_path)
        return model
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Error cargando modelo: {str(e)}",
        )


def get_model_metadata():
    """Obtiene metadata del modelo."""
    try:
        with open(settings.model_metadata_path, "r") as f:
            return json.load(f)
    except Exception as e:
        return {}


"""def validate_token(token: str) -> bool:
    # Para desarrollo, aceptar cualquier token
    # En producción, validar contra base de datos o servicio de autenticación
    if settings.api_debug:
        return True
    
    # Implementación básica de validación
    # En un caso real, usaríamos python-jose para validar JWT
    if not token:
        return False
    
    # Aquí iría la lógica real de validación
    # Por ahora, solo verificar que no esté vacío
    return len(token) > 10"""


def validate_token(token: str) -> bool:
    """
    Valida un token JWT.
    Para desarrollo, aceptamos tokens simples.
    """
    # EN DESARROLLO: Aceptar cualquier token que comience con "demo"
    if settings.api_debug:
        # Aceptar tokens comunes de desarrollo
        valid_tokens = ["demo_token", "test_token", "dev_token", "bearer_token"]
        return token in valid_tokens or token.startswith("demo_")

    # EN PRODUCCIÓN: Validación real con JWT
    # Por ahora, misma lógica simplificada
    return token.startswith("demo_") or len(token) > 10


def get_current_user(token: str):
    """
    Obtiene el usuario actual basado en el token.
    """
    if not validate_token(token):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Token inválido"
        )

    # En producción, decodificar el token JWT para obtener usuario
    return {"username": "demo_user", "role": "user"}
