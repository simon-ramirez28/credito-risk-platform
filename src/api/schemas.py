"""
Esquemas Pydantic para la API.
"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


class CreditApplication(BaseModel):
    """Esquema para una solicitud de crédito individual."""

    # Información personal
    edad: int = Field(..., ge=18, le=100, description="Edad del solicitante")
    genero: str = Field(..., description="Género (M/F)")
    estado_civil: str = Field(..., description="Estado civil")
    dependientes: int = Field(..., ge=0, le=10, description="Número de dependientes")

    # Información financiera
    ingreso_mensual: float = Field(..., gt=0, description="Ingreso mensual")
    gastos_mensuales: float = Field(..., ge=0, description="Gastos mensuales")
    total_adeudado: float = Field(..., ge=0, description="Deuda total actual")
    ahorros: float = Field(..., ge=0, description="Ahorros totales")
    score_bancario: int = Field(
        ..., ge=300, le=850, description="Score bancario (300-850)"
    )

    # Información laboral
    antiguedad_empleo: int = Field(
        ..., ge=0, description="Antigüedad en meses en el empleo actual"
    )
    tipo_contrato: str = Field(..., description="Tipo de contrato laboral")

    # Historial crediticio
    num_creditos_previos: int = Field(
        ..., ge=0, description="Número de créditos previos"
    )
    max_dias_mora: int = Field(
        ..., ge=0, description="Máximo días de mora en créditos anteriores"
    )
    creditos_problematicos: int = Field(
        ..., ge=0, description="Créditos con problemas de pago"
    )

    # Información adicional
    tipo_vivienda: str = Field(..., description="Tipo de vivienda")
    antiguedad_residencia: int = Field(
        ..., ge=0, description="Años en la residencia actual"
    )

    class Config:
        schema_extra = {
            "example": {
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
                "antiguedad_residencia": 5,
            }
        }

    @validator("gastos_mensuales")
    def validate_expenses(cls, v, values):
        """Valida que los gastos no excedan el ingreso."""
        if "ingreso_mensual" in values and v > values["ingreso_mensual"] * 0.9:
            raise ValueError(
                "Los gastos mensuales no pueden exceder el 90% del ingreso"
            )
        return v


class PredictionResponse(BaseModel):
    """Respuesta de una predicción."""

    prediction: int = Field(..., description="0 = No default, 1 = Default")
    probability_default: float = Field(
        ..., ge=0, le=1, description="Probabilidad de default"
    )
    risk_score: int = Field(
        ..., ge=300, le=850, description="Score de riesgo (300-850)"
    )
    risk_category: str = Field(..., description="Categoría de riesgo")
    features_used: List[str] = Field(
        ..., description="Features utilizadas en la predicción"
    )
    message: str = Field(..., description="Mensaje descriptivo")


class BatchPredictionRequest(BaseModel):
    """Solicitud de predicciones por lotes."""

    applications: List[CreditApplication] = Field(
        ..., description="Lista de solicitudes"
    )


class BatchPredictionItem(BaseModel):
    """Item individual en respuesta de batch."""

    application_id: str
    prediction: Optional[int]
    probability_default: Optional[float]
    risk_score: Optional[int]
    risk_category: str
    status: str


class BatchPredictionResponse(BaseModel):
    """Respuesta de predicciones por lotes."""

    total_applications: int
    successful_predictions: int
    failed_predictions: int
    predictions: List[BatchPredictionItem]


class ModelInfo(BaseModel):
    """Información sobre el modelo cargado."""

    model_type: str
    model_name: str
    training_date: str
    feature_count: int
    metrics: Dict[str, Any]
    best_params: Dict[str, Any]


class HealthCheck(BaseModel):
    """Respuesta de health check."""

    status: str
    message: Optional[str] = None
    model_status: Optional[str] = None
    features_loaded: Optional[int] = None
    timestamp: Optional[str] = None


class ErrorResponse(BaseModel):
    """Respuesta de error estándar."""

    error: str
    detail: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
