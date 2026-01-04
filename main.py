"""
Punto de entrada principal para la plataforma de riesgo crediticio.
"""
import argparse
import logging
from pathlib import Path
from src.utils.data_generator import GeneradorDatosCrediticios
from src.etl.pipeline import PipelineETLCrediticio

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generar_datos(args):
    """Genera datos sintéticos."""
    logger.info(f"Generando {args.clientes} clientes...")
    
    generador = GeneradorDatosCrediticios(
        n_clientes=args.clientes,
        seed=args.seed
    )
    
    resultado = generador.generar_todos_datos(output_dir=args.output)
    
    logger.info(f"Datos generados exitosamente")
    logger.info(f"Default rate: {resultado['metadata']['default_rate']*100:.2f}%")
    
    return resultado


def ejecutar_pipeline(args):
    """Ejecuta el pipeline ETL."""
    logger.info("Ejecutando pipeline ETL...")
    
    pipeline = PipelineETLCrediticio(
        input_dir=args.input,
        output_dir=args.output
    )
    
    metadata = pipeline.run(formato_salida=args.format)
    
    logger.info(f"Pipeline completado exitosamente")
    logger.info(f"Registros procesados: {metadata['num_registros']}")
    
    return metadata


def main():
    """Función principal."""
    parser = argparse.ArgumentParser(
        description='Plataforma de Análisis de Riesgo Crediticio - Fase 1'
    )
    
    subparsers = parser.add_subparsers(dest='comando', help='Comando a ejecutar')
    
    # Comando para generar datos
    parser_generar = subparsers.add_parser('generar', help='Generar datos sintéticos')
    parser_generar.add_argument('--clientes', type=int, default=1000, help='Número de clientes')
    parser_generar.add_argument('--output', type=str, default='data/raw', help='Directorio de salida')
    parser_generar.add_argument('--seed', type=int, default=42, help='Semilla aleatoria')
    
    # Comando para ejecutar pipeline
    parser_pipeline = subparsers.add_parser('pipeline', help='Ejecutar pipeline ETL')
    parser_pipeline.add_argument('--input', type=str, default='data/raw', help='Directorio de entrada')
    parser_pipeline.add_argument('--output', type=str, default='data/processed', help='Directorio de salida')
    parser_pipeline.add_argument('--format', type=str, choices=['parquet', 'csv', 'json'], 
                               default='parquet', help='Formato de salida')
    
    # Comando para todo
    parser_todo = subparsers.add_parser('todo', help='Generar datos y ejecutar pipeline')
    parser_todo.add_argument('--clientes', type=int, default=1000, help='Número de clientes')
    parser_todo.add_argument('--seed', type=int, default=42, help='Semilla aleatoria')
    
    args = parser.parse_args()
    
    # Crear directorios base
    Path('data/raw').mkdir(parents=True, exist_ok=True)
    Path('data/processed').mkdir(parents=True, exist_ok=True)
    
    # Ejecutar comando
    if args.comando == 'generar':
        generar_datos(args)
        
    elif args.comando == 'pipeline':
        ejecutar_pipeline(args)
        
    elif args.comando == 'todo':
        # Primero generar datos
        gen_args = argparse.Namespace(
            clientes=args.clientes,
            output='data/raw',
            seed=args.seed
        )
        generar_datos(gen_args)
        
        # Luego ejecutar pipeline
        pipe_args = argparse.Namespace(
            input='data/raw',
            output='data/processed',
            format='parquet'
        )
        ejecutar_pipeline(pipe_args)
        
    else:
        parser.print_help()


if __name__ == '__main__':
    main()