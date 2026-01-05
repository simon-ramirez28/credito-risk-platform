"""
Pruebas para modelos de ML.
"""
import pytest
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from src.models.train_model import CreditRiskModel


def create_test_data(n_samples=100, n_features=10):
    """Crea datos de prueba para entrenamiento."""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=5,
        n_redundant=2,
        n_classes=2,
        weights=[0.7, 0.3],  # Clases desbalanceadas
        random_state=42
    )
    
    feature_names = [f'feature_{i}' for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    df['default'] = y
    
    return df


def test_model_initialization():
    """Prueba la inicialización de diferentes tipos de modelos."""
    # Random Forest
    model_rf = CreditRiskModel(model_type='random_forest', random_state=42)
    assert model_rf.model_type == 'random_forest'
    assert model_rf.model is not None
    
    # Gradient Boosting
    model_gb = CreditRiskModel(model_type='gradient_boosting', random_state=42)
    assert model_gb.model_type == 'gradient_boosting'
    assert model_gb.model is not None
    
    # Logistic Regression
    model_lr = CreditRiskModel(model_type='logistic', random_state=42)
    assert model_lr.model_type == 'logistic'
    assert model_lr.model is not None


def test_prepare_data():
    """Prueba la preparación de datos."""
    df = create_test_data(n_samples=100, n_features=5)
    model = CreditRiskModel()
    
    X_train, X_test, y_train, y_test = model.prepare_data(df, test_size=0.2)
    
    # Verificar shapes
    assert X_train.shape[0] == 80  # 80% de 100
    assert X_test.shape[0] == 20   # 20% de 100
    assert len(y_train) == 80
    assert len(y_test) == 20
    
    # Verificar que se guardaron los nombres de features
    assert model.feature_names is not None
    assert len(model.feature_names) == 5


def test_train_and_evaluate():
    """Prueba el entrenamiento y evaluación del modelo."""
    df = create_test_data(n_samples=200, n_features=8)
    model = CreditRiskModel(model_type='random_forest', random_state=42)
    
    # Preparar datos
    X_train, X_test, y_train, y_test = model.prepare_data(df, test_size=0.2)
    
    # Entrenar
    train_metrics = model.train(X_train, y_train, use_cv=False)
    assert 'train' in model.metrics
    assert 'roc_auc' in model.metrics['train']
    
    # Evaluar
    test_metrics = model.evaluate(X_test, y_test)
    assert 'test' in model.metrics
    assert 'roc_auc' in model.metrics['test']
    
    # Verificar que el modelo fue entrenado
    assert model.model is not None
    assert hasattr(model.model, 'predict')


def test_get_feature_importance():
    """Prueba la obtención de importancia de features."""
    df = create_test_data(n_samples=100, n_features=5)
    model = CreditRiskModel(model_type='random_forest')
    
    X_train, X_test, y_train, y_test = model.prepare_data(df, test_size=0.2)
    model.train(X_train, y_train, use_cv=False)
    
    importance = model.get_feature_importance()
    
    # Verificar estructura
    assert isinstance(importance, pd.DataFrame)
    assert 'feature' in importance.columns
    assert 'importance' in importance.columns
    
    if not importance.empty:
        assert len(importance) == len(model.feature_names)


def test_save_model(tmp_path):
    """Prueba el guardado del modelo."""
    df = create_test_data(n_samples=50, n_features=3)
    model = CreditRiskModel()
    
    X_train, X_test, y_train, y_test = model.prepare_data(df, test_size=0.2)
    model.train(X_train, y_train, use_cv=False)
    
    # Guardar en directorio temporal
    saved_files = model.save_model(output_dir=str(tmp_path))
    
    # Verificar que se crearon los archivos
    assert 'model_path' in saved_files
    assert 'metadata_path' in saved_files
    
    import joblib
    import json
    
    # Verificar que el modelo se puede cargar
    loaded_model = joblib.load(saved_files['model_path'])
    assert loaded_model is not None
    
    # Verificar metadata
    with open(saved_files['metadata_path'], 'r') as f:
        metadata = json.load(f)
    
    assert 'model_type' in metadata
    assert 'feature_names' in metadata


if __name__ == '__main__':
    pytest.main([__file__, '-v'])