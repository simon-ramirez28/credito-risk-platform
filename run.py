#!/usr/bin/env python3
"""
Script para ejecutar diferentes componentes del sistema.
"""
import argparse
import subprocess
import sys
from pathlib import Path

def run_api():
    """Ejecuta la API."""
    print("🚀 Iniciando API...")
    subprocess.run([
        "uvicorn", "src.api.app:app",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--reload"
    ])

def run_dashboard():
    """Ejecuta el dashboard."""
    print("📊 Iniciando Dashboard...")
    subprocess.run([
        "streamlit", "run", "src/dashboard/app.py",
        "--server.port", "8501",
        "--server.address", "0.0.0.0"
    ])

def run_tests():
    """Ejecuta todos los tests."""
    print("🧪 Ejecutando tests...")
    subprocess.run(["pytest", "tests/", "-v"])

def run_pipeline():
    """Ejecuta el pipeline completo."""
    print("⚙️ Ejecutando pipeline completo...")
    from main import main as pipeline_main
    sys.argv = ["main.py", "todo", "--clientes", "1000"]
    pipeline_main()

def main():
    """Función principal."""
    parser = argparse.ArgumentParser(description="Sistema de Riesgo Crediticio")
    
    subparsers = parser.add_subparsers(dest="command", help="Comando a ejecutar")
    
    # API
    parser_api = subparsers.add_parser("api", help="Iniciar API")
    
    # Dashboard
    parser_dashboard = subparsers.add_parser("dashboard", help="Iniciar Dashboard")
    
    # Tests
    parser_tests = subparsers.add_parser("test", help="Ejecutar tests")
    
    # Pipeline
    parser_pipeline = subparsers.add_parser("pipeline", help="Ejecutar pipeline completo")
    
    # All
    parser_all = subparsers.add_parser("all", help="Iniciar todo (API + Dashboard)")
    
    args = parser.parse_args()
    
    if args.command == "api":
        run_api()
    elif args.command == "dashboard":
        run_dashboard()
    elif args.command == "test":
        run_tests()
    elif args.command == "pipeline":
        run_pipeline()
    elif args.command == "all":
        # En producción, usaríamos algo como supervisor
        print("⚠️  Para ejecutar múltiples servicios, usa Docker Compose")
        print("   o ejecuta en terminales separadas:")
        print("   python run.py api")
        print("   python run.py dashboard")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()