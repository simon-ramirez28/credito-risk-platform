"""
API REST para predicciones de riesgo crediticio.
"""
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import joblib
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
import logging
from loguru import logger

from .config import settings
from .schemas import (
    CreditApplication, PredictionResponse, 
    BatchPredictionRequest, BatchPredictionResponse,
    ModelInfo, HealthCheck
)
from .dependencies import get_model, validate_token

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger.add(
    settings.logs_dir / "api.log",
    rotation="500 MB",
    retention="10 days",
    level="INFO"
)

# Security
security = HTTPBearer()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan manager para cargar recursos al inicio y limpiar al final.
    """
    # Cargar modelo al inicio
    try:
        logger.info("Cargando modelo...")
        app.state.model = joblib.load(settings.model_path)
        
        # Cargar metadata del modelo
        with open(settings.model_metadata_path, 'r') as f:
            app.state.model_metadata = json.load(f)
        
        # Cargar feature names
        app.state.feature_names = app.state.model_metadata.get('feature_names', [])
        
        logger.info(f"Modelo cargado: {settings.model_path}")
        logger.info(f"Features: {len(app.state.feature_names)}")
        
    except Exception as e:
        logger.error(f"Error cargando modelo: {e}")
        app.state.model = None
        app.state.model_metadata = {}
        app.state.feature_names = []
    
    yield
    
    # Limpiar al final (si es necesario)
    logger.info("Apagando API...")

# Crear aplicación FastAPI
app = FastAPI(
    title="API de Riesgo Crediticio",
    description="API para predicciones de riesgo crediticio usando ML",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, especificar dominios
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Middleware para logging de requests
@app.middleware("http")
async def log_requests(request, call_next):
    logger.info(f"Request: {request.method} {request.url}")
    response = await call_next(request)
    logger.info(f"Response: {response.status_code}")
    return response


@app.get("/", response_model=HealthCheck)
async def root():
    """
    Endpoint raíz para verificar que la API está funcionando.
    """
    return {
        "status": "healthy",
        "message": "API de Riesgo Crediticio funcionando",
        "version": "1.0.0"
    }


@app.get("/health", response_model=HealthCheck)
async def health_check():
    """
    Health check endpoint para monitoreo.
    """
    model_status = "loaded" if app.state.model is not None else "error"
    
    return {
        "status": "healthy",
        "model_status": model_status,
        "features_loaded": len(app.state.feature_names),
        "timestamp": pd.Timestamp.now().isoformat()
    }


@app.get("/model/info", response_model=ModelInfo)
async def get_model_info():
    """
    Obtiene información sobre el modelo cargado.
    """
    if app.state.model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Modelo no cargado"
        )
    
    metadata = app.state.model_metadata
    
    return {
        "model_type": metadata.get("model_type", "unknown"),
        "model_name": metadata.get("model_name", "unknown"),
        "training_date": metadata.get("timestamp", "unknown"),
        "feature_count": metadata.get("feature_count", 0),
        "metrics": metadata.get("metrics", {}),
        "best_params": metadata.get("best_params", {})
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    application: CreditApplication,
    token: HTTPAuthorizationCredentials = Depends(security)
):
    """
    Realiza una predicción de riesgo para una solicitud de crédito.
    
    Requiere autenticación con token.
    """
    # Validar token (simplificado para desarrollo)
    # En producción, implementar autenticación real
    if not validate_token(token.credentials):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token inválido o expirado"
        )
    
    if app.state.model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Modelo no disponible"
        )
    
    try:
        # Convertir solicitud a DataFrame
        input_data = application.dict()
        
        # Preparar datos para el modelo
        df = pd.DataFrame([input_data])
        
        # Aplicar transformaciones necesarias
        df = prepare_input_data(df, app.state.feature_names)
        
        # Verificar que tenemos todas las features necesarias
        missing_features = set(app.state.feature_names) - set(df.columns)
        if missing_features:
            logger.warning(f"Features faltantes: {missing_features}")
            # Rellenar features faltantes con valores por defecto
            for feature in missing_features:
                df[feature] = 0
        
        # Ordenar features en el orden esperado por el modelo
        df = df[app.state.feature_names]
        
        # Realizar predicción
        prediction = app.state.model.predict(df)
        probability = app.state.model.predict_proba(df)
        
        # Calcular score de riesgo (0-1000)
        risk_score = calculate_risk_score(probability[0][1])
        risk_category = categorize_risk(risk_score)
        
        return PredictionResponse(
            prediction=int(prediction[0]),
            probability_default=float(probability[0][1]),
            risk_score=risk_score,
            risk_category=risk_category,
            features_used=app.state.feature_names[:10],  # Mostrar solo primeras 10
            message="Predicción completada exitosamente"
        )
        
    except Exception as e:
        logger.error(f"Error en predicción: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error procesando la solicitud: {str(e)}"
        )


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(
    batch_request: BatchPredictionRequest,
    token: HTTPAuthorizationCredentials = Depends(security)
):
    """
    Realiza predicciones por lotes para múltiples solicitudes.
    """
    if not validate_token(token.credentials):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token inválido o expirado"
        )
    
    if app.state.model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Modelo no disponible"
        )
    
    try:
        applications = batch_request.applications
        predictions = []
        
        for i, app_data in enumerate(applications):
            try:
                # Preparar datos
                df = pd.DataFrame([app_data.dict()])
                df = prepare_input_data(df, app.state.feature_names)
                
                # Rellenar features faltantes
                missing_features = set(app.state.feature_names) - set(df.columns)
                for feature in missing_features:
                    df[feature] = 0
                
                df = df[app.state.feature_names]
                
                # Predecir
                prediction = app.state.model.predict(df)
                probability = app.state.model.predict_proba(df)
                risk_score = calculate_risk_score(probability[0][1])
                risk_category = categorize_risk(risk_score)
                
                predictions.append({
                    "application_id": f"app_{i:04d}",
                    "prediction": int(prediction[0]),
                    "probability_default": float(probability[0][1]),
                    "risk_score": risk_score,
                    "risk_category": risk_category,
                    "status": "success"
                })
                
            except Exception as e:
                predictions.append({
                    "application_id": f"app_{i:04d}",
                    "prediction": None,
                    "probability_default": None,
                    "risk_score": None,
                    "risk_category": "ERROR",
                    "status": f"error: {str(e)[:100]}"
                })
        
        return BatchPredictionResponse(
            total_applications=len(applications),
            successful_predictions=sum(1 for p in predictions if p["status"] == "success"),
            failed_predictions=sum(1 for p in predictions if p["status"] != "success"),
            predictions=predictions
        )
        
    except Exception as e:
        logger.error(f"Error en predicción por lotes: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error procesando el lote: {str(e)}"
        )


@app.get("/features")
async def get_features():
    """
    Obtiene la lista de features que espera el modelo.
    """
    if app.state.model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Modelo no cargado"
        )
    
    return {
        "features": app.state.feature_names,
        "count": len(app.state.feature_names),
        "sample_input": generate_sample_input(app.state.feature_names)
    }


def prepare_input_data(df: pd.DataFrame, feature_names: list) -> pd.DataFrame:
    """
    Prepara los datos de entrada aplicando transformaciones necesarias.
    """
    df_processed = df.copy()
    
    # Aplicar transformaciones básicas
    # 1. Calcular ratios si faltan
    if 'ingreso_mensual' in df_processed.columns and 'gastos_mensuales' in df_processed.columns:
        if 'capacidad_pago' not in df_processed.columns:
            df_processed['capacidad_pago'] = df_processed['ingreso_mensual'] - df_processed['gastos_mensuales']
    
    # 2. Calcular edad si no está presente
    if 'fecha_nacimiento' in df_processed.columns and 'edad' not in df_processed.columns:
        # Asumir formato YYYY-MM-DD
        try:
            birth_dates = pd.to_datetime(df_processed['fecha_nacimiento'])
            today = pd.Timestamp.now()
            df_processed['edad'] = (today - birth_dates).dt.days // 365
        except:
            df_processed['edad'] = 35  # Valor por defecto
    
    # 3. Codificar variables categóricas básicas
    categorical_mappings = {
        'genero': {'M': 0, 'F': 1},
        'estado_civil': {'soltero': 0, 'casado': 1, 'divorciado': 2, 'viudo': 3}
    }
    
    for col, mapping in categorical_mappings.items():
        if col in df_processed.columns:
            df_processed[col] = df_processed[col].map(mapping).fillna(0)
    
    return df_processed


def calculate_risk_score(probability: float) -> int:
    """
    Calcula un score de riesgo de 300 a 850 (similar a FICO).
    """
    # Mapear probabilidad (0-1) a score (300-850)
    # Probabilidad más baja = score más alto (mejor)
    base_score = 850
    risk_penalty = probability * 550  # Máxima penalización de 550 puntos
    
    score = int(base_score - risk_penalty)
    
    # Asegurar que está en el rango correcto
    return max(300, min(850, score))


def categorize_risk(score: int) -> str:
    """
    Categoriza el riesgo basado en el score.
    """
    if score >= 750:
        return "EXCELENTE"
    elif score >= 700:
        return "BUENO"
    elif score >= 650:
        return "REGULAR"
    elif score >= 600:
        return "DEFICIENTE"
    else:
        return "POBRE"


def generate_sample_input(feature_names: list) -> Dict[str, Any]:
    """
    Genera un ejemplo de input para la API.
    """
    sample = {}
    
    # Mapeo de tipos de datos para features comunes
    feature_types = {
        'edad': 35,
        'ingreso_mensual': 3000.0,
        'gastos_mensuales': 2000.0,
        'total_adeudado': 5000.0,
        'score_bancario': 700,
        'antiguedad_empleo': 24,
        'dependientes': 1,
        'genero': 'M',
        'estado_civil': 'casado'
    }
    
    for feature in feature_names[:10]:  # Solo primeras 10 para ejemplo
        for key, value in feature_types.items():
            if key in feature.lower():
                sample[feature] = value
                break
        else:
            # Valor por defecto
            if 'ratio' in feature.lower() or 'rate' in feature.lower():
                sample[feature] = 0.5
            elif any(word in feature.lower() for word in ['score', 'total', 'count', 'num']):
                sample[feature] = 0
            else:
                sample[feature] = 0.0
    
    return sample


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.app:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload,
        log_level="info" if settings.api_debug else "warning"
    )