"""
Pruebas para el pipeline ETL.
"""
import pytest
import pandas as pd
import numpy as np
from src.etl.pipeline import PipelineETLCrediticio


def crear_datos_prueba():
    """Crea datos de prueba para las pruebas."""
    datos = {
        'demografico': pd.DataFrame({
            'cliente_id': [f'CLI{i:06d}' for i in range(5)],
            'nombre': [f'Cliente {i}' for i in range(5)],
            'edad': [25, 30, 35, 40, 45],
            'genero': ['M', 'F', 'M', 'F', 'M'],
            'estado_civil': ['soltero', 'casado', 'divorciado', 'casado', 'soltero'],
            'dependientes': [0, 2, 1, 3, 0],
            'nivel_educacion': ['universitaria', 'media', 'postgrado', 'universitaria', 'tecnica'],
            'tipo_vivienda': ['propia', 'arrendada', 'propia', 'hipotecada', 'familiar'],
            'ciudad': ['Ciudad A', 'Ciudad B', 'Ciudad A', 'Ciudad C', 'Ciudad B'],
            'antiguedad_residencia': [2, 5, 10, 3, 1],
            'telefono': ['123', '456', '789', '012', '345'],
            'email': ['a@test.com', 'b@test.com', 'c@test.com', 'd@test.com', 'e@test.com'],
            'fecha_registro': pd.date_range('2020-01-01', periods=5)
        }),
        
        'financiero': pd.DataFrame({
            'cliente_id': [f'CLI{i:06d}' for i in range(5)],
            'ingreso_mensual': [3000, 2500, 4000, 3500, 2000],
            'ingreso_anual': [36000, 30000, 48000, 42000, 24000],
            'fuente_ingreso': ['empleo', 'empleo', 'independiente', 'empleo', 'pension'],
            'antiguedad_empleo': [24, 12, 36, 6, 0],
            'tipo_contrato': ['indefinido', 'temporal', 'independiente', 'indefinido', 'sin_contrato'],
            'empresa_tamano': ['grande', 'mediana', 'N/A', 'pequena', 'N/A'],
            'otros_ingresos': [500, 0, 1000, 200, 300],
            'gastos_mensuales': [2000, 1800, 3000, 2500, 1500],
            'ahorros': [5000, 2000, 10000, 3000, 1000],
            'score_bancario': [750, 650, 800, 700, 550]
        }),
        
        'historial': pd.DataFrame({
            'cliente_id': ['CLI000000', 'CLI000000', 'CLI000001', 'CLI000002', 'CLI000003'],
            'credito_id': ['CRD000000-000', 'CRD000000-001', 'CRD000001-000', 'CRD000002-000', 'CRD000003-000'],
            'monto': [1000, 2000, 5000, 3000, 4000],
            'plazo_meses': [12, 24, 36, 12, 24],
            'tasa_interes': [0.12, 0.15, 0.10, 0.18, 0.20],
            'fecha_inicio': pd.date_range('2020-01-01', periods=5),
            'fecha_fin': pd.date_range('2021-01-01', periods=5),
            'estado': ['pagado', 'moroso', 'pagado', 'incumplido', 'pagado'],
            'dias_mora': [0, 60, 0, 120, 0],
            'tipo_credito': ['personal', 'automotriz', 'hipotecario', 'personal', 'tarjeta'],
            'institucion': ['BancoA', 'BancoB', 'BancoA', 'FinancieraC', 'BancoB']
        }),
        
        'target': pd.DataFrame({
            'cliente_id': [f'CLI{i:06d}' for i in range(5)],
            'default': [0, 1, 0, 1, 0],
            'score_riesgo': [0.2, 0.8, 0.3, 0.7, 0.4],
            'probabilidad_default': [20.0, 80.0, 30.0, 70.0, 40.0],
            'categoria_riesgo': ['BAJO', 'MUY_ALTO', 'MODERADO', 'ALTO', 'MODERADO']
        })
    }
    
    return datos


def test_pipeline_transform():
    """Prueba la transformación del pipeline."""
    datos = crear_datos_prueba()
    
    pipeline = PipelineETLCrediticio()
    df_transformado = pipeline.transform(datos)
    
    # Verificar estructura básica
    assert isinstance(df_transformado, pd.DataFrame)
    assert len(df_transformado) == 5  # 5 clientes
    
    # Verificar que se unieron correctamente
    assert 'default' in df_transformado.columns
    assert 'total_adeudado' in df_transformado.columns
    
    # Verificar que se crearon features
    assert 'deuda_ingreso_ratio' in df_transformado.columns
    assert 'empleo_estable' in df_transformado.columns
    
    # Verificar que se eliminaron columnas sensibles
    assert 'telefono' not in df_transformado.columns
    assert 'email' not in df_transformado.columns


def test_pipeline_reporte_calidad():
    """Prueba la generación de reporte de calidad."""
    datos = crear_datos_prueba()
    
    pipeline = PipelineETLCrediticio()
    df_transformado = pipeline.transform(datos)
    reporte = pipeline._generar_reporte_calidad(df_transformado)
    
    assert 'registros_totales' in reporte
    assert 'registros_nulos' in reporte
    assert 'estadisticas_numericas' in reporte
    assert reporte['registros_totales'] == 5


if __name__ == '__main__':
    pytest.main([__file__, '-v'])