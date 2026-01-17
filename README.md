# 🏦 Plataforma de Análisis de Riesgo Crediticio

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104%2B-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red)
![Docker](https://img.shields.io/badge/Docker-20.10%2B-2496ED)
![GitHub Actions](https://img.shields.io/badge/GitHub_Actions-✔-2088FF)

Sistema completo para análisis y predicción de riesgo crediticio usando Machine Learning, con API REST, Dashboard interactivo y pipeline CI/CD automatizado.

## ✨ Características Principales

### 🎯 **Machine Learning**
- **Generación de datos sintéticos** realistas para pruebas
- **Pipeline ETL** completo y reproducible
- **Feature engineering** avanzado con dominio financiero
- **Múltiples algoritmos**: Random Forest, Gradient Boosting, Logistic Regression
- **Validación cruzada** y ajuste de hiperparámetros
- **Métricas completas**: ROC AUC, precisión, recall, F1-score

### 🚀 **API REST con FastAPI**
- **Documentación automática** (Swagger/OpenAPI)
- **Autenticación** con tokens JWT
- **Predicciones individuales** y **por lotes**
- **Health checks** y monitoreo
- **Rate limiting** y manejo de errores
- **Caching** con Redis

### 📊 **Dashboard Interactivo**
- **Visualizaciones en tiempo real** con Plotly
- **Formularios interactivos** para predicciones
- **Carga de archivos CSV** para procesamiento por lotes
- **Análisis exploratorio** integrado
- **Segmentación** por variables demográficas
- **Generación de reportes** automáticos

### 🐳 **Infraestructura Moderna**
- **Dockerización completa** con multi-stage builds
- **Orquestación** con Docker Compose
- **CI/CD automático** con GitHub Actions
- **Base de datos PostgreSQL** para producción
- **Redis** para caching y colas
- **Nginx** como reverse proxy

## 🚀 Comenzando

### Prerrequisitos
- **Python 3.10+**
- **Docker 20.10+** (opcional, recomendado)
- **Docker Compose 2.0+** (opcional)
- **Git**

### Instalación Rápida

#### Método 1: Docker (Recomendado)

```bash
# 1. Clonar el repositorio
git clone https://github.com/tu-usuario/credito-risk-platform.git
cd credito-risk-platform

# 2. Iniciar todos los servicios
make docker-up

# 3. Verificar que todo funciona
curl http://localhost:8000/health

# 4. Abrir el dashboard
# Navegador: http://localhost:8501
```

#### Método 2: Desarrollo Local

```bash
# 1. Clonar el repositorio
git clone https://github.com/tu-usuario/credito-risk-platform.git
cd credito-risk-platform

# 2. Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 3. Instalar dependencias
make install

# 4. Generar datos y entrenar modelo
make generate-data
make train-model

# 5. Iniciar servicios (en terminales separadas)
make run-api      # Terminal 1 - API en http://localhost:8000
make run-dashboard # Terminal 2 - Dashboard en http://localhost:8501
```

## 📁 Estructura del Proyecto

```
credito-risk-platform/
├── src/                    # Código fuente
│   ├── api/               # API REST con FastAPI
│   │   ├── app.py         # Aplicación principal
│   │   ├── schemas.py     # Esquemas Pydantic
│   │   ├── config.py      # Configuración
│   │   └── dependencies.py # Dependencias e inyección
│   ├── dashboard/         # Dashboard con Streamlit
│   │   ├── app.py         # Aplicación principal
│   │   └── config.py      # Configuración
│   ├── etl/              # Pipeline ETL
│   │   └── pipeline.py    # Extracción, transformación, carga
│   ├── features/          # Feature engineering
│   │   └── feature_engineering.py # Ingeniería de características
│   ├── models/           # Modelos de Machine Learning
│   │   └── train_model.py # Entrenamiento y predicción
│   ├── validation/       # Validación de modelos
│   │   └── model_validator.py # Validación y monitoreo
│   └── utils/            # Utilidades
│       └── data_generator.py # Generador de datos sintéticos
├── docker/               # Configuración Docker
│   ├── Dockerfile.api    # Imagen para API
│   ├── Dockerfile.dashboard # Imagen para Dashboard
│   ├── Dockerfile.train  # Imagen para entrenamiento
│   ├── docker-compose.yml # Orquestación
│   └── nginx/            # Configuración Nginx
│       └── nginx.conf
├── tests/               # Tests unitarios e integración
│   ├── test_api.py      # Tests para API
│   ├── test_dashboard.py # Tests para Dashboard
│   ├── test_features.py  # Tests para feature engineering
│   ├── test_models.py   # Tests para modelos ML
│   ├── test_generator.py # Tests para generador
│   └── test_pipeline.py # Tests para pipeline ETL
├── scripts/             # Scripts utilitarios
│   ├── entrypoint.sh    # Script de entrada
│   ├── wait-for-it.sh   # Espera por servicios
│   └── healthcheck.sh   # Health checks
├── data/                # Datos (gitignored)
│   ├── raw/            # Datos crudos
│   ├── processed/      # Datos procesados
│   └── features/       # Features para ML
├── models/             # Modelos entrenados (gitignored)
├── notebooks/          # Jupyter notebooks para análisis
│   └── exploratory_analysis.ipynb
├── .github/            # GitHub Actions CI/CD
│   └── workflows/
│       ├── ci.yml      # Integración continua
│       ├── cd.yml      # Despliegue continuo
│       └── train-model.yml # Entrenamiento automático
├── docs/               # Documentación adicional
├── requirements/       # Dependencias organizadas
│   ├── api.txt        # Dependencias API
│   ├── dashboard.txt  # Dependencias Dashboard
│   └── dev.txt        # Dependencias desarrollo
├── Makefile           # Comandos automatizados
├── main.py            # Punto de entrada principal
└── README.md          # Este archivo
```

## 🐳 Docker

### Servicios Disponibles

| Servicio | Puerto | Descripción | URL |
|----------|--------|-------------|-----|
| **API** | 8000 | API REST FastAPI | http://localhost:8000 |
| **Dashboard** | 8501 | Dashboard Streamlit | http://localhost:8501 |
| **PostgreSQL** | 5432 | Base de datos | localhost:5432 |
| **Redis** | 6379 | Cache y colas | localhost:6379 |
| **Nginx** | 80 | Reverse proxy | http://localhost |

### Comandos Docker

```bash
# Construir todas las imágenes
make docker-build

# Iniciar todos los servicios (en background)
make docker-up

# Ver logs en tiempo real
make docker-logs

# Detener todos los servicios
make docker-down

# Entrenar modelo en contenedor
make docker-train

# Acceder a shell del contenedor API
make docker-shell

# Ver estado de los servicios
docker-compose ps

# Reconstruir y reiniciar un servicio específico
docker-compose up -d --build api
```

### Configuración de Variables de Entorno

Crea un archivo `.env` en la raíz del proyecto:

```env
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_DEBUG=false
API_AUTH_TOKEN=your-secure-token-here
SECRET_KEY=your-secret-key-change-in-production
ALGORITHM=HS256

# Database
DATABASE_URL=postgresql://credit_user:credit_password@postgres:5432/credit_db

# Redis
REDIS_URL=redis://redis:6379/0

# Model
MODEL_PATH=models/random_forest_model_latest.pkl
MODEL_METADATA_PATH=models/random_forest_model_latest_metadata.json

# Dashboard
DASHBOARD_PORT=8501
API_URL=http://api:8000  # Interno para Docker
```

## 🔧 Desarrollo

### Instalación para Desarrollo

```bash
# Clonar y configurar
git clone https://github.com/tu-usuario/credito-risk-platform.git
cd credito-risk-platform

# Instalar dependencias completas
make install

# Ejecutar pipeline completo
python main.py todo --clientes 1000

# Ejecutar tests
make test

# Verificar calidad de código
make lint

# Formatear código automáticamente
make format
```

### Comandos Principales

```bash
# Pipeline completo (datos + features + modelo)
python main.py todo --clientes 5000

# Solo generación de datos
python main.py fase1 --clientes 1000

# Solo feature engineering
python main.py fase2-features

# Solo entrenamiento de modelo
python main.py fase2-train --model-type random_forest --tune-hyperparams

# Solo validación
python main.py fase2-validate

# Iniciar solo API
python run.py api

# Iniciar solo Dashboard
python run.py dashboard
```

### Tests

```bash
# Ejecutar todos los tests
pytest tests/ -v

# Ejecutar tests con cobertura
pytest tests/ --cov=src --cov-report=html

# Tests específicos
pytest tests/test_api.py -v
pytest tests/test_models.py -v

# Tests con mayor detalle
pytest tests/ -v --tb=short
```

## 📊 Uso del Sistema

### API REST

La API está disponible en `http://localhost:8000`.

#### Endpoints Principales

| Método | Endpoint | Descripción |
|--------|----------|-------------|
| `GET` | `/` | Health check básico |
| `GET` | `/health` | Health check detallado |
| `GET` | `/model/info` | Información del modelo cargado |
| `GET` | `/features` | Lista de features esperadas |
| `POST` | `/predict` | Predicción individual |
| `POST` | `/predict/batch` | Predicción por lotes |

#### Ejemplo de Predicción

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Authorization: Bearer your-token" \
  -H "Content-Type: application/json" \
  -d '{
    "edad": 35,
    "genero": "M",
    "estado_civil": "casado",
    "dependientes": 1,
    "ingreso_mensual": 3000.0,
    "gastos_mensuales": 2000.0,
    "total_adeudado": 5000.0,
    "ahorros": 10000.0,
    "score_bancario": 720,
    "antiguedad_empleo": 24,
    "tipo_contrato": "indefinido",
    "num_creditos_previos": 2,
    "max_dias_mora": 15,
    "creditos_problematicos": 0,
    "tipo_vivienda": "propia",
    "antiguedad_residencia": 5
  }'
```

#### Respuesta de Ejemplo

```json
{
  "prediction": 0,
  "probability_default": 0.1234,
  "risk_score": 785,
  "risk_category": "EXCELENTE",
  "features_used": ["edad", "ingreso_mensual", ...],
  "message": "Predicción completada exitosamente"
}
```

### Dashboard

Disponible en `http://localhost:8501`

#### Funcionalidades del Dashboard

1. **📈 Visión General**
   - KPIs del sistema
   - Distribución de edades e ingresos
   - Mapa de calor de correlaciones

2. **🎯 Predicciones**
   - Formulario interactivo para predicciones individuales
   - Carga de archivos CSV para predicciones por lotes
   - Visualización de resultados con gráficos

3. **📊 Análisis**
   - Segmentación por variables demográficas
   - Detección de patrones y correlaciones
   - Generación de reportes automáticos

4. **⚙️ Configuración**
   - Ajuste de parámetros del sistema
   - Configuración de conexiones API

## 🤖 Machine Learning

### Características del Modelo

#### Features Generadas

| Categoría | Ejemplos de Features |
|-----------|---------------------|
| **Demográficas** | Edad, género, estado civil, dependientes |
| **Financieras** | Ingreso mensual, gastos, ahorros, deuda total |
| **Laborales** | Antigüedad empleo, tipo contrato, estabilidad |
| **Crediticias** | Historial de créditos, días de mora, score bancario |
| **Calculadas** | Ratio deuda/ingreso, capacidad de pago, score compuesto |

#### Algoritmos Disponibles

1. **Random Forest** (por defecto)
   - Robustez a outliers
   - Feature importance automática
   - Buen performance con datos no lineales

2. **Gradient Boosting**
   - Alta precisión
   - Manejo de relaciones complejas
   - Requiere más ajuste de hiperparámetros

3. **Logistic Regression**
   - Interpretabilidad
   - Rapidez de entrenamiento
   - Baseline para comparación

#### Métricas de Evaluación

- **Accuracy**: Exactitud general
- **Precision**: Exactitud en predicciones positivas
- **Recall**: Sensibilidad para detectar defaults
- **F1-Score**: Balance entre precisión y recall
- **ROC AUC**: Capacidad discriminativa del modelo
- **Confusion Matrix**: Visualización de errores

### Pipeline de Entrenamiento

```python
# Ejemplo de entrenamiento programático
from src.models.train_model import CreditRiskModel
import pandas as pd

# Cargar datos
df = pd.read_parquet('data/features/features_engineered.parquet')

# Inicializar modelo
model = CreditRiskModel(model_type='random_forest')

# Preparar datos (train/test split)
X_train, X_test, y_train, y_test = model.prepare_data(df)

# Entrenar con cross-validation
model.train(X_train, y_train, use_cv=True)

# Evaluar en test
model.evaluate(X_test, y_test)

# Guardar modelo
model.save_model('models/')
```

## 🔄 CI/CD Pipeline

### GitHub Actions Workflows

#### 1. **CI (Continuous Integration)**
- Ejecuta en cada push y pull request
- Tests unitarios con pytest
- Linting con flake8 y black
- Type checking con mypy
- Build de imágenes Docker

#### 2. **CD (Continuous Deployment)**
- Ejecuta en releases y manualmente
- Despliegue automático a servidor
- Migraciones de base de datos
- Health checks post-deployment
- Notificaciones a Slack

#### 3. **Entrenamiento de Modelos**
- Ejecución programada (semanal)
- Descarga de datos actualizados
- Entrenamiento de nuevo modelo
- Upload a almacenamiento en la nube
- Notificaciones de actualización

### Configuración de Secrets

Para que CI/CD funcione, configura estos secrets en GitHub:

| Secret | Descripción |
|--------|-------------|
| `DOCKER_USERNAME` | Usuario de Docker Hub |
| `DOCKER_PASSWORD` | Contraseña de Docker Hub |
| `SSH_PRIVATE_KEY` | Clave SSH para despliegue |
| `SERVER_HOST` | Host del servidor de producción |
| `SERVER_USER` | Usuario del servidor |
| `PRODUCTION_DATABASE_URL` | URL de base de datos producción |
| `SLACK_WEBHOOK_URL` | Webhook para notificaciones |

## 🌐 Despliegue en la Nube

### Opción 1: Railway (Más Simple)

```bash
# 1. Instalar CLI de Railway
npm i -g @railway/cli

# 2. Iniciar proyecto
railway init

# 3. Desplegar
railway up
```

### Opción 2: Render

1. Conectar repositorio de GitHub
2. Crear servicio Web Service
3. Configurar:
   - Build Command: `docker build -f docker/Dockerfile.api .`
   - Start Command: `python src/api/app.py`
   - Port: 8000

### Opción 3: AWS ECS

```bash
# 1. Crear ECR repository
aws ecr create-repository --repository-name credit-risk-api

# 2. Construir y subir imagen
docker build -f docker/Dockerfile.api -t credit-risk-api .
docker tag credit-risk-api:latest <account-id>.dkr.ecr.<region>.amazonaws.com/credit-risk-api:latest
docker push <account-id>.dkr.ecr.<region>.amazonaws.com/credit-risk-api:latest

# 3. Crear task definition y service
aws ecs create-service --cluster credit-cluster --service-name credit-service --task-definition credit-task
```

### Opción 4: Google Cloud Run

```bash
# 1. Construir imagen
docker build -f docker/Dockerfile.api -t gcr.io/<project-id>/credit-api .

# 2. Subir a Google Container Registry
docker push gcr.io/<project-id>/credit-api

# 3. Desplegar en Cloud Run
gcloud run deploy credit-api --image gcr.io/<project-id>/credit-api --platform managed
```

## 🔍 Monitoreo y Logging

### Health Checks

```bash
# Verificar salud de la API
curl http://localhost:8000/health

# Verificar salud del Dashboard
curl http://localhost:8501/_stcore/health

# Verificar servicios Docker
docker-compose ps
```

### Logs

```bash
# Ver logs de todos los servicios
make docker-logs

# Ver logs específicos
docker-compose logs api
docker-compose logs dashboard

# Seguir logs en tiempo real
docker-compose logs -f api

# Ver logs con timestamps
docker-compose logs --timestamps
```

### Métricas

La API expone métricas en formato JSON:

```bash
# Obtener información del modelo
curl http://localhost:8000/model/info | jq '.metrics'
```

## 🛠️ Troubleshooting

### Problemas Comunes y Soluciones

#### 1. API no responde
```bash
# Verificar que el contenedor está corriendo
docker ps | grep api

# Ver logs de error
docker-compose logs api

# Reiniciar servicio
docker-compose restart api
```

#### 2. Dashboard no carga
```bash
# Verificar conexión con API
curl http://api:8000/health

# Verificar puerto
netstat -tulpn | grep 8501

# Limpiar cache de Streamlit
rm -rf ~/.streamlit
```

#### 3. Error de conexión a base de datos
```bash
# Verificar que PostgreSQL está corriendo
docker-compose ps postgres

# Probar conexión manual
docker-compose exec postgres psql -U credit_user -d credit_db

# Reiniciar servicios dependientes
docker-compose restart api dashboard
```

#### 4. Modelo no encontrado
```bash
# Verificar que existe el archivo
ls -la models/*.pkl

# Entrenar nuevo modelo
make docker-train

# Copiar modelo manualmente
cp models/random_forest_model_*.pkl models/random_forest_model_latest.pkl
```

#### 5. Memory issues
```bash
# Limpiar cache de Docker
docker system prune -f

# Limpiar volúmenes no usados
docker volume prune -f

# Ver uso de recursos
docker stats
```

## 📈 Roadmap y Mejoras Futuras

### 🚀 Próximas Características

1. **Autenticación Avanzada**
   - OAuth2 con proveedores externos
   - Roles y permisos
   - Refresh tokens

2. **Más Modelos de ML**
   - XGBoost y LightGBM
   - Deep Learning con TensorFlow
   - Ensemble methods

3. **Monitorización Avanzada**
   - Prometheus + Grafana
   - Alertas automáticas
   - Dashboard de métricas

4. **Integraciones**
   - Webhooks para notificaciones
   - API para terceros
   - Exportación a BI tools

5. **Escalabilidad**
   - Kubernetes para orquestación
   - Auto-scaling
   - Load balancing

### 🎯 Optimizaciones Técnicas

- [ ] Cache avanzado con Redis
- [ ] Async database operations
- [ ] Background jobs con Celery
- [ ] API versioning
- [ ] Rate limiting por usuario
- [ ] Circuit breakers para dependencias

## 🤝 Contribución

¡Las contribuciones son bienvenidas! Por favor sigue estos pasos:

1. **Fork** el repositorio
2. **Crea una rama** para tu feature (`git checkout -b feature/AmazingFeature`)
3. **Commit** tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. **Push** a la rama (`git push origin feature/AmazingFeature`)
5. **Abre un Pull Request**

### Guías de Estilo

- **Código**: Sigue PEP 8, usa Black para formateo
- **Documentación**: Usa Google-style docstrings
- **Commits**: Mensajes descriptivos en inglés
- **Tests**: Escribe tests para nuevas funcionalidades

### Estructura de Commits

```
feat: nueva funcionalidad
fix: corrección de bug
docs: cambios en documentación
style: formato, puntos y coma, etc (sin cambios funcionales)
refactor: refactorización de código
test: añadir o corregir tests
chore: cambios en build, config, etc
```

## 📄 Licencia

Este proyecto está licenciado bajo la **MIT License** - ver el archivo [LICENSE](LICENSE) para más detalles.

## 🙏 Agradecimientos

- **[FastAPI](https://fastapi.tiangolo.com/)** - Para la API rápida y moderna
- **[Streamlit](https://streamlit.io/)** - Para el dashboard interactivo
- **[Scikit-learn](https://scikit-learn.org/)** - Para los algoritmos de ML
- **[Docker](https://www.docker.com/)** - Para la containerización
- **[Plotly](https://plotly.com/)** - Para las visualizaciones
- **[Pandas](https://pandas.pydata.org/)** - Para el procesamiento de datos

## 📞 Soporte

Para soporte, por favor:

1. **Revisa la documentación** y troubleshooting guide
2. **Busca issues existentes** en GitHub
3. **Abre un nuevo issue** si no encuentras solución

**Contacto**: [ramirezdata22@gmail.com](mailto:ramirezdata22@gmail.com)

**Discusiones**: [GitHub Discussions](https://github.com/simon-ramirez28/credito-risk-platform/discussions)

---

<div align="center">
  
**Hecho con ❤️ para la comunidad de Data Engineering**

[⭐ Da una estrella en GitHub](https://github.com/simon-ramirez28/credito-risk-platform)

</div>
