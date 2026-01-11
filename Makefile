.PHONY: help install test api dashboard pipeline all docker-up docker-down clean

# Colores
GREEN=\033[0;32m
YELLOW=\033[1;33m
RED=\033[0;31m
NC=\033[0m # No Color

help:
	@echo "Comandos disponibles:"
	@echo "  ${GREEN}install${NC}     - Instalar dependencias"
	@echo "  ${GREEN}test${NC}        - Ejecutar tests"
	@echo "  ${GREEN}api${NC}         - Iniciar API"
	@echo "  ${GREEN}dashboard${NC}   - Iniciar Dashboard"
	@echo "  ${GREEN}pipeline${NC}    - Ejecutar pipeline completo"
	@echo "  ${GREEN}all${NC}         - Iniciar API y Dashboard"
	@echo "  ${GREEN}docker-up${NC}   - Iniciar con Docker Compose"
	@echo "  ${GREEN}docker-down${NC} - Detener Docker Compose"
	@echo "  ${GREEN}clean${NC}       - Limpiar archivos temporales"

install:
	@echo "${YELLOW}Instalando dependencias...${NC}"
	pip install -r requirements.txt
	@echo "${GREEN}✅ Dependencias instaladas${NC}"

test:
	@echo "${YELLOW}Ejecutando tests...${NC}"
	pytest tests/ -v --cov=src --cov-report=html
	@echo "${GREEN}✅ Tests completados${NC}"

api:
	@echo "${YELLOW}Iniciando API...${NC}"
	uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload

dashboard:
	@echo "${YELLOW}Iniciando Dashboard...${NC}"
	streamlit run src/dashboard/app.py --server.port 8501 --server.address 0.0.0.0

pipeline:
	@echo "${YELLOW}Ejecutando pipeline...${NC}"
	python main.py todo --clientes 1000
	@echo "${GREEN}✅ Pipeline completado${NC}"

all:
	@echo "${YELLOW}Iniciando todos los servicios...${NC}"
	@echo "${RED}⚠️  Ejecutar en terminales separadas:${NC}"
	@echo "  make api"
	@echo "  make dashboard"

docker-up:
	@echo "${YELLOW}Iniciando con Docker Compose...${NC}"
	docker-compose up --build

docker-down:
	@echo "${YELLOW}Deteniendo Docker Compose...${NC}"
	docker-compose down

clean:
	@echo "${YELLOW}Limpiando archivos temporales...${NC}"
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name "htmlcov" -exec rm -rf {} +
	@echo "${GREEN}✅ Limpieza completada${NC}"