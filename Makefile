.PHONY: help install test lint format clean docker-build docker-up docker-down docker-logs docker-train deploy

# Colores
GREEN=\033[0;32m
YELLOW=\033[1;33m
RED=\033[0;31m
BLUE=\033[0;34m
NC=\033[0m # No Color

# Variables
DOCKER_COMPOSE = docker-compose -f docker/docker-compose.yml
APP_NAME = credit-risk-platform

help:
	@echo "${BLUE}${APP_NAME} - Comandos disponibles:${NC}"
	@echo ""
	@echo "${GREEN}Desarrollo:${NC}"
	@echo "  ${YELLOW}install${NC}      - Instalar dependencias"
	@echo "  ${YELLOW}test${NC}         - Ejecutar tests"
	@echo "  ${YELLOW}lint${NC}         - Verificar código con linters"
	@echo "  ${YELLOW}format${NC}       - Formatear código automáticamente"
	@echo "  ${YELLOW}run-api${NC}      - Ejecutar API localmente"
	@echo "  ${YELLOW}run-dashboard${NC} - Ejecutar Dashboard localmente"
	@echo ""
	@echo "${GREEN}Docker:${NC}"
	@echo "  ${YELLOW}docker-build${NC} - Construir imágenes Docker"
	@echo "  ${YELLOW}docker-up${NC}    - Iniciar servicios con Docker Compose"
	@echo "  ${YELLOW}docker-down${NC}  - Detener servicios Docker"
	@echo "  ${YELLOW}docker-logs${NC}  - Ver logs de servicios"
	@echo "  ${YELLOW}docker-train${NC} - Entrenar modelo en contenedor"
	@echo "  ${YELLOW}docker-shell${NC} - Abrir shell en contenedor API"
	@echo ""
	@echo "${GREEN}Utilidades:${NC}"
	@echo "  ${YELLOW}clean${NC}        - Limpiar archivos temporales"
	@echo "  ${YELLOW}generate-data${NC} - Generar datos sintéticos"
	@echo "  ${YELLOW}train-model${NC}  - Entrenar nuevo modelo"
	@echo "  ${YELLOW}deploy${NC}       - Desplegar a producción"
	@echo ""

# Desarrollo
install:
	@echo "${YELLOW}Instalando dependencias...${NC}"
	pip install -r requirements.txt
	pip install -r requirements/dev.txt
	@echo "${GREEN}✅ Dependencias instaladas${NC}"

test:
	@echo "${YELLOW}Ejecutando tests...${NC}"
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term-missing

lint:
	@echo "${YELLOW}Ejecutando linters...${NC}"
	flake8 src/ --count --max-complexity=10 --max-line-length=127 --statistics
	mypy src/ --ignore-missing-imports
	black --check src/

format:
	@echo "${YELLOW}Formateando código...${NC}"
	black src/
	isort src/

run-api:
	@echo "${YELLOW}Iniciando API...${NC}"
	uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload

run-dashboard:
	@echo "${YELLOW}Iniciando Dashboard...${NC}"
	streamlit run src/dashboard/app.py --server.port 8501 --server.address 0.0.0.0

# Docker
docker-build:
	@echo "${YELLOW}Construyendo imágenes Docker...${NC}"
	$(DOCKER_COMPOSE) build --no-cache

docker-up:
	@echo "${YELLOW}Iniciando servicios con Docker Compose...${NC}"
	$(DOCKER_COMPOSE) up -d
	@echo "${GREEN}✅ Servicios iniciados${NC}"
	@echo ""
	@echo "${BLUE}📊 Servicios disponibles:${NC}"
	@echo "  API:          http://localhost:8000"
	@echo "  API Docs:     http://localhost:8000/docs"
	@echo "  Dashboard:    http://localhost:8501"
	@echo "  PostgreSQL:   localhost:5432"
	@echo "  Redis:        localhost:6379"
	@echo ""

docker-down:
	@echo "${YELLOW}Deteniendo servicios Docker...${NC}"
	$(DOCKER_COMPOSE) down -v

docker-logs:
	@echo "${YELLOW}Mostrando logs...${NC}"
	$(DOCKER_COMPOSE) logs -f

docker-train:
	@echo "${YELLOW}Entrenando modelo en contenedor...${NC}"
	$(DOCKER_COMPOSE) run --rm train fase2-train --model-type random_forest

docker-shell:
	@echo "${YELLOW}Abriendo shell en contenedor API...${NC}"
	$(DOCKER_COMPOSE) exec api bash

# Utilidades
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
	find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} +
	rm -rf build/ dist/ *.egg-info
	@echo "${GREEN}✅ Limpieza completada${NC}"

generate-data:
	@echo "${YELLOW}Generando datos sintéticos...${NC}"
	python main.py fase1 --clientes 5000
	@echo "${GREEN}✅ Datos generados${NC}"

train-model:
	@echo "${YELLOW}Entrenando modelo...${NC}"
	python main.py fase2-train --model-type random_forest --tune-hyperparams
	@echo "${GREEN}✅ Modelo entrenado${NC}"

deploy:
	@echo "${YELLOW}Desplegando a producción...${NC}"
	@echo "${RED}⚠️  Esta acción requiere configuración previa${NC}"
	@echo ""
	@echo "Pasos manuales:"
	@echo "1. Asegurar que Docker y docker-compose están instalados en el servidor"
	@echo "2. Configurar variables de entorno en el servidor"
	@echo "3. Ejecutar: make docker-up en el servidor"
	@echo ""
	@echo "Para despliegue automático, configura GitHub Actions"

# Comando para desarrollo rápido
dev: install test lint
	@echo "${GREEN}✅ Desarrollo configurado${NC}"

# Comando para producción
prod: docker-build docker-up
	@echo "${GREEN}✅ Producción desplegada${NC}"