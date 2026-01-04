"""
Pipeline ETL básico para datos crediticios.
"""
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from pathlib import Path
import json
from typing import Dict, Any, Optional

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PipelineETLCrediticio:
    """Pipeline ETL para procesamiento de datos crediticios."""
    
    def __init__(self, input_dir: str = 'data/raw', output_dir: str = 'data/processed'):
        """
        Inicializa el pipeline.
        
        Args:
            input_dir: Directorio con datos crudos
            output_dir: Directorio para datos procesados
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        
        # Crear directorios si no existen
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Pipeline inicializado: {input_dir} -> {output_dir}")
    
    def extract(self, archivos: Optional[Dict[str, str]] = None) -> Dict[str, pd.DataFrame]:
        """
        Extrae datos de archivos CSV.
        
        Args:
            archivos: Diccionario con nombres de archivos específicos
                     Si es None, usa los más recientes
        
        Returns:
            Diccionario con DataFrames extraídos
        """
        logger.info("Iniciando extracción de datos...")
        
        if archivos is None:
            # Encontrar archivos más recientes
            archivos = self._encontrar_archivos_recientes()
        
        datos = {}
        for nombre, archivo in archivos.items():
            try:
                ruta = self.input_dir / archivo
                if ruta.exists():
                    datos[nombre] = pd.read_csv(ruta)
                    logger.info(f"  ✅ {nombre}: {len(datos[nombre])} registros")
                else:
                    logger.warning(f"  ⚠️ Archivo no encontrado: {ruta}")
                    datos[nombre] = pd.DataFrame()
            except Exception as e:
                logger.error(f"  ❌ Error cargando {archivo}: {e}")
                datos[nombre] = pd.DataFrame()
        
        logger.info("Extracción completada")
        return datos
    
    def _encontrar_archivos_recientes(self) -> Dict[str, str]:
        """Encuentra los archivos más recientes en el directorio."""
        archivos = {}
        
        # Buscar archivos con patrones conocidos
        patrones = {
            'demografico': '*demograficos*.csv',
            'financiero': '*financieros*.csv',
            'historial': '*historial*.csv',
            'target': '*target*.csv'
        }
        
        for nombre, patron in patrones.items():
            archivos_encontrados = list(self.input_dir.glob(patron))
            if archivos_encontrados:
                # Ordenar por fecha de modificación (más reciente primero)
                archivos_encontrados.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                archivos[nombre] = archivos_encontrados[0].name
        
        return archivos
    
    def transform(self, datos: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Transforma y une los datos.
        
        Args:
            datos: Diccionario con DataFrames extraídos
        
        Returns:
            DataFrame unido y transformado
        """
        logger.info("Iniciando transformación de datos...")
        
        # Verificar que tenemos todos los datos necesarios
        if any(df.empty for df in datos.values()):
            logger.error("Faltan datos para la transformación")
            raise ValueError("No se pueden transformar datos incompletos")
        
        # 1. Unir datos demográficos y financieros
        logger.info("  Uniendo datos demográficos y financieros...")
        df_clientes = pd.merge(
            datos['demografico'],
            datos['financiero'],
            on='cliente_id',
            how='inner'
        )
        
        # 2. Agregar target
        logger.info("  Añadiendo variable target...")
        df_clientes = pd.merge(
            df_clientes,
            datos['target'],
            on='cliente_id',
            how='left'
        )
        
        # 3. Agregaciones del historial crediticio
        logger.info("  Procesando historial crediticio...")
        df_historial = datos['historial'].copy()
        
        # Calcular métricas por cliente
        historial_agg = df_historial.groupby('cliente_id').agg({
            'monto': ['sum', 'mean', 'count'],
            'dias_mora': ['max', 'mean', 'sum'],
            'estado': lambda x: (x.isin(['moroso', 'incumplido'])).sum(),
            'tasa_interes': 'mean'
        }).reset_index()
        
        # Aplanar columnas multi-nivel
        historial_agg.columns = [
            'cliente_id',
            'total_adeudado', 'promedio_credito', 'num_creditos_previos',
            'max_dias_mora', 'promedio_dias_mora', 'total_dias_mora',
            'creditos_problematicos', 'tasa_interes_promedio'
        ]
        
        # 4. Unir con datos de clientes
        logger.info("  Uniendo historial crediticio...")
        df_completo = pd.merge(
            df_clientes,
            historial_agg,
            on='cliente_id',
            how='left'
        )
        
        # 5. Transformaciones y limpieza
        logger.info("  Aplicando transformaciones...")
        df_completo = self._aplicar_transformaciones(df_completo)
        
        # 6. Feature engineering básico (más en Fase 2)
        logger.info("  Creando features básicas...")
        df_completo = self._crear_features_basicas(df_completo)
        
        logger.info(f"Transformación completada: {len(df_completo)} registros")
        return df_completo
    
    def _aplicar_transformaciones(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aplica transformaciones de limpieza y normalización."""
        df_transformado = df.copy()
        
        # 1. Imputar valores nulos
        columnas_numericas = df_transformado.select_dtypes(include=[np.number]).columns
        
        for col in columnas_numericas:
            if df_transformado[col].isnull().any():
                # Para métricas de crédito, asumir 0 si no hay historial
                if 'credito' in col.lower() or 'mora' in col.lower():
                    df_transformado[col] = df_transformado[col].fillna(0)
                else:
                    # Para otras, usar la mediana
                    df_transformado[col] = df_transformado[col].fillna(df_transformado[col].median())
        
        # 2. Codificar variables categóricas básicas
        if 'genero' in df_transformado.columns:
            df_transformado['genero_encoded'] = df_transformado['genero'].map({'M': 0, 'F': 1})
        
        if 'estado_civil' in df_transformado.columns:
            # One-hot encoding básico
            estados_dummies = pd.get_dummies(df_transformado['estado_civil'], prefix='estado')
            df_transformado = pd.concat([df_transformado, estados_dummies], axis=1)
        
        # 3. Eliminar columnas innecesarias para análisis
        columnas_a_eliminar = [
            'nombre', 'telefono', 'email', 'fecha_registro',
            'institucion'  # Si existe del historial
        ]
        
        for col in columnas_a_eliminar:
            if col in df_transformado.columns:
                df_transformado = df_transformado.drop(columns=[col])
        
        return df_transformado
    
    def _crear_features_basicas(self, df: pd.DataFrame) -> pd.DataFrame:
        """Crea features básicas para análisis."""
        df_features = df.copy()
        
        # 1. Ratios financieros
        if 'ingreso_mensual' in df_features.columns:
            if 'total_adeudado' in df_features.columns:
                df_features['deuda_ingreso_ratio'] = np.where(
                    df_features['ingreso_mensual'] > 0,
                    df_features['total_adeudado'] / df_features['ingreso_mensual'],
                    0
                )
            
            if 'gastos_mensuales' in df_features.columns:
                df_features['ahorro_mensual'] = df_features['ingreso_mensual'] - df_features['gastos_mensuales']
                df_features['gastos_ingreso_ratio'] = np.where(
                    df_features['ingreso_mensual'] > 0,
                    df_features['gastos_mensuales'] / df_features['ingreso_mensual'],
                    0
                )
        
        # 2. Features de estabilidad
        if 'antiguedad_empleo' in df_features.columns:
            df_features['empleo_estable'] = (df_features['antiguedad_empleo'] >= 12).astype(int)
        
        if 'antiguedad_residencia' in df_features.columns:
            df_features['residencia_estable'] = (df_features['antiguedad_residencia'] >= 2).astype(int)
        
        # 3. Features de riesgo crediticio
        if 'num_creditos_previos' in df_features.columns:
            df_features['tiene_historial'] = (df_features['num_creditos_previos'] > 0).astype(int)
            df_features['frecuencia_creditos'] = np.where(
                df_features['edad'] > 18,
                df_features['num_creditos_previos'] / (df_features['edad'] - 18),
                0
            )
        
        if 'creditos_problematicos' in df_features.columns:
            df_features['tiene_problemas'] = (df_features['creditos_problematicos'] > 0).astype(int)
        
        # 4. Features demográficas combinadas
        if 'edad' in df_features.columns and 'dependientes' in df_features.columns:
            df_features['edad_dependientes_ratio'] = np.where(
                df_features['dependientes'] > 0,
                df_features['edad'] / df_features['dependientes'],
                0
            )
        
        # 5. Score compuesto simple
        score_factors = []
        
        if 'deuda_ingreso_ratio' in df_features.columns:
            # Menor ratio = mejor
            deuda_score = np.where(df_features['deuda_ingreso_ratio'] < 0.5, 1.0, 0.5)
            score_factors.append(deuda_score)
        
        if 'empleo_estable' in df_features.columns:
            score_factors.append(df_features['empleo_estable'])
        
        if 'score_bancario' in df_features.columns:
            # Normalizar score bancario (300-850 -> 0-1)
            bancario_score = (df_features['score_bancario'] - 300) / 550
            score_factors.append(bancario_score)
        
        if score_factors:
            df_features['score_riesgo_calculado'] = np.mean(score_factors, axis=0)
        
        return df_features
    
    def load(self, df: pd.DataFrame, formato: str = 'parquet') -> Dict[str, Any]:
        """
        Carga los datos transformados.
        
        Args:
            df: DataFrame transformado
            formato: Formato de salida ('parquet', 'csv', 'json')
        
        Returns:
            Metadata del proceso
        """
        logger.info("Iniciando carga de datos...")
        
        fecha_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Guardar en el formato especificado
        archivo_base = f'clientes_procesados_{fecha_str}'
        
        if formato == 'parquet':
            archivo = self.output_dir / f'{archivo_base}.parquet'
            df.to_parquet(archivo, index=False)
        elif formato == 'csv':
            archivo = self.output_dir / f'{archivo_base}.csv'
            df.to_csv(archivo, index=False)
        elif formato == 'json':
            archivo = self.output_dir / f'{archivo_base}.json'
            df.to_json(archivo, orient='records', indent=2)
        else:
            raise ValueError(f"Formato no soportado: {formato}")
        
        # Generar reporte de calidad de datos
        reporte = self._generar_reporte_calidad(df)
        
        # Guardar metadata
        metadata = {
            'fecha_procesamiento': fecha_str,
            'archivo_salida': str(archivo),
            'formato': formato,
            'num_registros': len(df),
            'num_columnas': len(df.columns),
            'columnas': df.columns.tolist(),
            'tipos_datos': df.dtypes.astype(str).to_dict(),
            'reporte_calidad': reporte,
            'default_rate': float(df['default'].mean()) if 'default' in df.columns else None
        }
        
        metadata_file = self.output_dir / f'metadata_{fecha_str}.json'
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Datos cargados en: {archivo}")
        logger.info(f"Metadata guardada en: {metadata_file}")
        
        return metadata
    
    def _generar_reporte_calidad(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Genera un reporte de calidad de datos."""
        reporte = {
            'registros_totales': len(df),
            'registros_nulos': {},
            'valores_unicos': {},
            'estadisticas_numericas': {}
        }
        
        # Analizar cada columna
        for col in df.columns:
            # Valores nulos
            nulos = df[col].isnull().sum()
            reporte['registros_nulos'][col] = {
                'total': int(nulos),
                'porcentaje': float(nulos / len(df) * 100)
            }
            
            # Valores únicos
            if df[col].dtype == 'object' or df[col].nunique() < 20:
                unicos = df[col].nunique()
                reporte['valores_unicos'][col] = {
                    'total': int(unicos),
                    'valores': df[col].unique().tolist() if unicos <= 10 else 'más de 10 valores'
                }
            
            # Estadísticas para columnas numéricas
            if pd.api.types.is_numeric_dtype(df[col]):
                reporte['estadisticas_numericas'][col] = {
                    'min': float(df[col].min()),
                    'max': float(df[col].max()),
                    'media': float(df[col].mean()),
                    'mediana': float(df[col].median()),
                    'std': float(df[col].std())
                }
        
        return reporte
    
    def run(self, formato_salida: str = 'parquet') -> Dict[str, Any]:
        """
        Ejecuta el pipeline completo.
        
        Args:
            formato_salida: Formato de archivo de salida
        
        Returns:
            Metadata del proceso
        """
        logger.info("[-] Iniciando ejecución del pipeline ETL")
        
        try:
            # 1. Extraer
            datos = self.extract()
            
            # 2. Transformar
            df_transformado = self.transform(datos)
            
            # 3. Cargar
            metadata = self.load(df_transformado, formato=formato_salida)
            
            logger.info("[ :D ] Pipeline ETL completado exitosamente")
            return metadata
            
        except Exception as e:
            logger.error(f"[X] Error en el pipeline ETL: {e}")
            raise


# Función principal para ejecutar desde línea de comandos
def main():
    """Función principal para ejecutar el pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Pipeline ETL para datos crediticios')
    parser.add_argument('--input', type=str, default='data/raw', help='Directorio de entrada')
    parser.add_argument('--output', type=str, default='data/processed', help='Directorio de salida')
    parser.add_argument('--format', type=str, choices=['parquet', 'csv', 'json'], 
                       default='parquet', help='Formato de salida')
    
    args = parser.parse_args()
    
    # Ejecutar pipeline
    pipeline = PipelineETLCrediticio(input_dir=args.input, output_dir=args.output)
    metadata = pipeline.run(formato_salida=args.format)
    
    print("\n📊 Resumen del procesamiento:")
    print(f"  Registros procesados: {metadata['num_registros']}")
    print(f"  Columnas generadas: {metadata['num_columnas']}")
    print(f"  Archivo generado: {metadata['archivo_salida']}")
    
    if metadata['default_rate'] is not None:
        print(f"  Tasa de default: {metadata['default_rate']*100:.2f}%")


if __name__ == '__main__':
    main()