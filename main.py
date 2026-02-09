"""
Punto de entrada principal para la plataforma de riesgo crediticio.
Fase 2: Feature Engineering + Modelo Simple
"""
import argparse
import logging
from pathlib import Path
import sys
import pandas as pd

# Añadir src al path
sys.path.append(str(Path(__file__).parent / 'src'))

from utils.data_generator import GeneradorDatosCrediticios
from etl.pipeline import PipelineETLCrediticio
from features.feature_engineering import CreditFeatureEngineer
from models.train_model import CreditRiskModel
from validation.model_validator import ModelValidator

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def fase1_generar_datos(args):
    """Fase 1: Generar datos sintéticos."""
    logger.info("=" * 60)
    logger.info("FASE 1: GENERACIÓN DE DATOS SINTÉTICOS")
    logger.info("=" * 60)
    
    generador = GeneradorDatosCrediticios(
        n_clientes=args.clientes,
        seed=args.seed
    )
    
    resultado = generador.generar_todos_datos(output_dir='data/raw')
    
    logger.info(f"->Datos generados: {args.clientes} clientes")
    logger.info(f"   Default rate: {resultado['metadata']['default_rate']*100:.2f}%")
    
    return resultado


def fase1_ejecutar_pipeline(args):
    """Fase 1: Ejecutar pipeline ETL básico."""
    logger.info("=" * 60)
    logger.info("FASE 1: PIPELINE ETL BÁSICO")
    logger.info("=" * 60)
    
    pipeline = PipelineETLCrediticio(
        input_dir='data/raw',
        output_dir='data/processed'
    )
    
    metadata = pipeline.run(formato_salida='parquet')
    
    logger.info(f"->Pipeline completado")
    logger.info(f"   Registros procesados: {metadata['num_registros']}")
    logger.info(f"   Columnas generadas: {metadata['num_columnas']}")
    
    return metadata


def fase2_feature_engineering(args):
    """Fase 2: Feature engineering avanzado."""
    logger.info("=" * 60)
    logger.info("FASE 2: FEATURE ENGINEERING AVANZADO")
    logger.info("=" * 60)
    
    # Encontrar archivo procesado más reciente
    processed_dir = Path('data/processed')
    parquet_files = list(processed_dir.glob('*.parquet'))
    
    if not parquet_files:
        logger.error("[X]  No se encontraron archivos procesados")
        logger.error("   Ejecuta primero: python main.py fase1")
        return None
    
    latest_file = max(parquet_files, key=lambda x: x.stat().st_mtime)
    logger.info(f"Cargando datos desde: {latest_file}")
    
    # Cargar datos
    df = pd.read_parquet(latest_file)
    
    # Feature engineering
    engineer = CreditFeatureEngineer(target_col='default')
    
    # Crear features avanzadas
    df_features = engineer.create_advanced_features(df)
    logger.info(f"Features creadas: {len(df_features.columns)} columnas")
    
    # Preparar para ML
    df_ml_ready = engineer.prepare_features_for_ml(df_features)
    logger.info(f"Preparado para ML: {len(df_ml_ready.columns)} columnas")
    
    # Guardar features
    features_dir = Path('data/features')
    features_dir.mkdir(exist_ok=True)
    
    features_file = features_dir / f'features_engineered_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.parquet'
    df_ml_ready.to_parquet(features_file, index=False)
    
    # Guardar metadata
    metadata_file = features_dir / f'features_metadata_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.json'
    engineer.save_feature_metadata(str(metadata_file))
    
    logger.info(f"->Features guardadas en: {features_file}")
    logger.info(f"   Metadata guardada en: {metadata_file}")
    
    return {
        'features_file': str(features_file),
        'metadata_file': str(metadata_file),
        'dataframe': df_ml_ready
    }


def fase2_entrenar_modelo(args):
    """Fase 2: Entrenar modelo de ML."""
    logger.info("=" * 60)
    logger.info("FASE 2: ENTRENAMIENTO DE MODELO")
    logger.info("=" * 60)
    
    import pandas as pd
    from models.train_model import CreditRiskModel
    
    # Encontrar archivo de features más reciente
    features_dir = Path('data/features')
    parquet_files = list(features_dir.glob('*.parquet'))
    
    if not parquet_files:
        logger.error("[X]  No se encontraron archivos de features")
        logger.error("   Ejecuta primero: python main.py fase2-features")
        return None
    
    latest_file = max(parquet_files, key=lambda x: x.stat().st_mtime)
    logger.info(f"Cargando features desde: {latest_file}")
    
    # Cargar datos
    df = pd.read_parquet(latest_file)
    
    # Entrenar modelo
    model = CreditRiskModel(
        model_type=args.model_type,
        random_state=args.random_state
    )
    
    # Preparar datos
    X_train, X_test, y_train, y_test = model.prepare_data(df, test_size=args.test_size)
    
    # Búsqueda de hiperparámetros si se solicita
    if args.tune_hyperparams:
        model.hyperparameter_tuning(X_train, y_train)
    
    # Entrenar
    model.train(X_train, y_train, use_cv=True)
    
    # Evaluar
    model.evaluate(X_test, y_test)
    
    # Guardar modelo
    output_dir = getattr(args, 'output_dir', 'models')
    saved_files = model.save_model(output_dir=output_dir)
    
    # Mostrar reporte
    print("\n" + model.create_model_report())
    
    logger.info(f"-> Modelo entrenado y guardado")
    logger.info(f"   Modelo: {saved_files['model_path']}")
    
    return saved_files


def fase2_validar_modelo(args):
    """Fase 2: Validar modelo."""
    logger.info("=" * 60)
    logger.info("FASE 2: VALIDACIÓN DE MODELO")
    logger.info("=" * 60)
    
    import pandas as pd
    from validation.model_validator import ModelValidator
    
    # Encontrar modelo más reciente
    models_dir = Path('models')
    model_files = list(models_dir.glob('*.pkl'))
    
    if not model_files:
        logger.error("[X]  No se encontraron modelos entrenados")
        logger.error("   Ejecuta primero: python main.py fase2-train")
        return None
    
    latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
    
    # Inferir metadata
    model_name = latest_model.stem
    metadata_files = list(models_dir.glob(f'*{model_name}*metadata*.json'))
    
    if not metadata_files:
        logger.error("[X]  No se encontró metadata del modelo")
        return None
    
    latest_metadata = max(metadata_files, key=lambda x: x.stat().st_mtime)
    
    # Cargar datos de validación
    logger.info(f"Cargando datos para validación...")
    if args.data:
        df = pd.read_parquet(args.data) if args.data.endswith('.parquet') else pd.read_csv(args.data)
    else:
        # Usar datos de features
        features_dir = Path('data/features')
        feature_files = list(features_dir.glob('*.parquet'))
        if feature_files:
            latest_features = max(feature_files, key=lambda x: x.stat().st_mtime)
            df = pd.read_parquet(latest_features)
        else:
            logger.error("[X]  No se encontraron datos para validación")
            return None
    
    # Validar
    validator = ModelValidator(str(latest_model), str(latest_metadata))
    validation_results = validator.validate_on_new_data(df)
    
    # Generar reporte
    report_path = validator.generate_validation_report(validation_results, 'reports')
    
    # Crear gráficos
    validator.create_validation_plots(validation_results, 'reports/plots')
    
    logger.info(f"->Validación completada")
    logger.info(f"   Reporte: {report_path}")
    
    return validation_results


def main():
    """Función principal."""
    parser = argparse.ArgumentParser(
        description='Plataforma de Análisis de Riesgo Crediticio - Fase 2',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  # Fase 1 completa
  python main.py fase1 --clientes 1000
  
  # Fase 2 completa
  python main.py fase2
  
  # Solo feature engineering
  python main.py fase2-features
  
  # Solo entrenamiento
  python main.py fase2-train --model-type random_forest
  
  # Solo validación
  python main.py fase2-validate
  
  # Todo el pipeline
  python main.py todo --clientes 1000
        """
    )
    
    subparsers = parser.add_subparsers(dest='comando', help='Comando a ejecutar')
    
    # FASE 1
    parser_fase1 = subparsers.add_parser('fase1', help='Ejecutar Fase 1 completa (generar datos + ETL)')
    parser_fase1.add_argument('--clientes', type=int, default=1000, help='Número de clientes')
    parser_fase1.add_argument('--seed', type=int, default=42, help='Semilla aleatoria')
    
    # FASE 2
    parser_fase2 = subparsers.add_parser('fase2', help='Ejecutar Fase 2 completa (features + modelo)')
    
    # Subcomandos Fase 2
    parser_fase2_features = subparsers.add_parser('fase2-features', help='Feature engineering avanzado')
    
    parser_fase2_train = subparsers.add_parser('fase2-train', help='Entrenar modelo de ML')
    parser_fase2_train.add_argument('--model-type', type=str, default='random_forest',
                                   choices=['random_forest', 'gradient_boosting', 'logistic'],
                                   help='Tipo de modelo')
    parser_fase2_train.add_argument('--test-size', type=float, default=0.2,
                                   help='Proporción para test split')
    parser_fase2_train.add_argument('--tune-hyperparams', action='store_true',
                                   help='Realizar búsqueda de hiperparámetros')
    parser_fase2_train.add_argument('--random-state', type=int, default=42,
                                   help='Semilla para reproducibilidad')
    parser_fase2_train.add_argument('--output-dir', type=str, default='models',
                                   help='Directorio donde guardar el modelo')
    
    parser_fase2_validate = subparsers.add_parser('fase2-validate', help='Validar modelo')
    parser_fase2_validate.add_argument('--data', type=str, help='Datos para validación (opcional)')
    
    # TODO
    parser_todo = subparsers.add_parser('todo', help='Ejecutar todo el pipeline (Fase 1 + Fase 2)')
    parser_todo.add_argument('--clientes', type=int, default=1000, help='Número de clientes')
    parser_todo.add_argument('--seed', type=int, default=42, help='Semilla aleatoria')
    
    args = parser.parse_args()
    
    # Crear directorios base
    Path('data/raw').mkdir(parents=True, exist_ok=True)
    Path('data/processed').mkdir(parents=True, exist_ok=True)
    Path('data/features').mkdir(parents=True, exist_ok=True)
    Path('models').mkdir(parents=True, exist_ok=True)
    Path('reports').mkdir(parents=True, exist_ok=True)
    Path('reports/plots').mkdir(parents=True, exist_ok=True)
    
    # Importar pandas aquí para no afectar otros comandos
    import pandas as pd
    
    # Ejecutar comando
    if args.comando == 'fase1':
        fase1_generar_datos(args)
        fase1_ejecutar_pipeline(args)
        
    elif args.comando == 'fase2':
        # Ejecutar Fase 2 completa
        fase2_feature_engineering(args)
        fase2_entrenar_modelo(args)
        fase2_validar_modelo(args)
        
    elif args.comando == 'fase2-features':
        fase2_feature_engineering(args)
        
    elif args.comando == 'fase2-train':
        fase2_entrenar_modelo(args)
        
    elif args.comando == 'fase2-validate':
        fase2_validar_modelo(args)
        
    elif args.comando == 'todo':
        # Fase 1
        fase1_generar_datos(args)
        fase1_ejecutar_pipeline(args)
        
        # Fase 2
        fase2_feature_engineering(args)
        fase2_entrenar_modelo(args)
        fase2_validar_modelo(args)
        
    else:
        parser.print_help()


if __name__ == '__main__':
    main()