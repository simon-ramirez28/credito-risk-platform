"""
Pruebas para el generador de datos.
"""
import pytest
import pandas as pd
from src.utils.data_generator import GeneradorDatosCrediticios


def test_generador_inicializacion():
    """Prueba la inicialización del generador."""
    generador = GeneradorDatosCrediticios(n_clientes=100, seed=42)
    assert generador.n_clientes == 100
    assert generador.seed == 42


def test_generar_datos_demograficos():
    """Prueba la generación de datos demográficos."""
    generador = GeneradorDatosCrediticios(n_clientes=10, seed=42)
    df = generador.generar_datos_demograficos()
    
    # Verificar estructura
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 10
    assert 'cliente_id' in df.columns
    assert 'edad' in df.columns
    assert 'genero' in df.columns
    
    # Verificar rangos
    assert df['edad'].between(18, 70).all()
    assert df['dependientes'].between(0, 5).all()


def test_generar_datos_financieros():
    """Prueba la generación de datos financieros."""
    generador = GeneradorDatosCrediticios(n_clientes=10, seed=42)
    df = generador.generar_datos_financieros()
    
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 10
    assert 'ingreso_mensual' in df.columns
    assert 'score_bancario' in df.columns
    
    # Verificar que el score bancario está en rango
    assert df['score_bancario'].between(300, 850).all()


def test_generar_todos_datos():
    """Prueba la generación completa de datos."""
    generador = GeneradorDatosCrediticios(n_clientes=5, seed=42)
    datos = generador.generar_todos_datos(output_dir='test_data')
    
    # Verificar que se generaron todos los DataFrames
    assert 'demografico' in datos
    assert 'financiero' in datos
    assert 'historial' in datos
    assert 'target' in datos
    assert 'metadata' in datos
    
    # Verificar que todos tienen el mismo número de clientes
    assert len(datos['demografico']) == 5
    assert len(datos['financiero']) == 5
    assert len(datos['target']) == 5
    
    # Limpiar archivos de prueba
    import shutil
    shutil.rmtree('test_data', ignore_errors=True)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])