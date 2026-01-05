"""
Pruebas para feature engineering.
"""
import pytest
import pandas as pd
import numpy as np
from src.features.feature_engineering import CreditFeatureEngineer


def test_feature_engineer_initialization():
    """Prueba la inicialización del feature engineer."""
    engineer = CreditFeatureEngineer(target_col='default', random_state=42)
    assert engineer.target_col == 'default'
    assert engineer.random_state == 42
    assert engineer.scaler is None
    assert engineer.imputer is None


def test_create_advanced_features():
    """Prueba la creación de features avanzadas."""
    # Crear datos de prueba
    df = pd.DataFrame({
        'edad': [25, 30, 35, 40, 45],
        'ingreso_mensual': [3000, 2500, 4000, 3500, 2000],
        'gastos_mensuales': [2000, 1800, 3000, 2500, 1500],
        'total_adeudado': [5000, 3000, 8000, 6000, 2000],
        'antiguedad_empleo': [12, 24, 6, 36, 0],
        'score_bancario': [750, 650, 800, 700, 550],
        'dependientes': [0, 2, 1, 3, 0]
    })
    
    engineer = CreditFeatureEngineer()
    df_features = engineer.create_advanced_features(df)
    
    # Verificar que se crearon nuevas features
    assert 'capacidad_pago' in df_features.columns
    assert 'deuda_ingreso_anual_ratio' in df_features.columns
    assert 'meses_liquidez' in df_features.columns
    assert 'composite_risk_score' in df_features.columns
    
    # Verificar que las features tienen valores razonables
    assert df_features['capacidad_pago'].min() >= 0
    assert df_features['deuda_ingreso_anual_ratio'].min() >= 0


def test_prepare_features_for_ml():
    """Prueba la preparación de features para ML."""
    # Crear datos de prueba con target
    df = pd.DataFrame({
        'edad': [25, 30, 35, 40, 45],
        'ingreso_mensual': [3000, 2500, 4000, 3500, 2000],
        'genero': ['M', 'F', 'M', 'F', 'M'],
        'default': [0, 1, 0, 1, 0]
    })
    
    engineer = CreditFeatureEngineer(target_col='default')
    df_ml_ready = engineer.prepare_features_for_ml(df)
    
    # Verificar que se procesaron las features
    assert engineer.feature_names is not None
    assert len(engineer.feature_names) > 0
    assert 'default' in df_ml_ready.columns
    
    # Verificar que las variables categóricas se procesaron
    assert 'genero' not in df_ml_ready.columns  # Debería estar codificada
    assert any('genero' in col for col in df_ml_ready.columns)  # O codificada como dummy


def test_get_feature_importance():
    """Prueba el cálculo de importancia de features."""
    df = pd.DataFrame({
        'feature1': np.random.randn(100),
        'feature2': np.random.randn(100),
        'feature3': np.random.randn(100),
        'default': np.random.randint(0, 2, 100)
    })
    
    engineer = CreditFeatureEngineer(target_col='default')
    importance = engineer.get_feature_importance(df)
    
    # Verificar estructura del resultado
    assert isinstance(importance, pd.DataFrame)
    assert 'feature' in importance.columns
    assert 'correlation_with_target' in importance.columns
    
    if not importance.empty:
        assert len(importance) == 3  # Tres features


if __name__ == '__main__':
    pytest.main([__file__, '-v'])