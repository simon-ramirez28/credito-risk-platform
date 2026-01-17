# 🏦 Plataforma de Análisis de Riesgo Crediticio

Sistema completo para análisis y predicción de riesgo crediticio usando Machine Learning, con API REST y Dashboard interactivo.

## 🚀 Características

- **Generación de datos sintéticos** realistas
- **Pipeline ETL** completo y reproducible
- **Feature engineering** avanzado
- **Modelos de ML** (Random Forest, Gradient Boosting, Logistic Regression)
- **API REST** con FastAPI para predicciones en tiempo real
- **Dashboard interactivo** con Streamlit
- **Dockerización** completa
- **CI/CD** con GitHub Actions
- **Despliegue** en múltiples entornos

## 📋 Prerrequisitos

- Python 3.10+
- Docker 20.10+ (opcional, para containerización)
- Docker Compose 2.0+ (opcional)
- Git

## 🛠️ Instalación Rápida

### Método 1: Desarrollo Local

```bash
# 1. Clonar repositorio
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

# 5. Iniciar servicios
make run-api          # Terminal 1
make run-dashboard    # Terminal 2
```

### Método 2: Docker (Recomendado para producción)

```bash
# 1. Clonar repositorio
git clone https://github.com/tu-usuario/credito-risk-platform.git
cd credito-risk-platform

# 2. Construir y ejecutar con Docker Compose
make docker-up

# 3. Verificar servicios
curl http://localhost:8000/health
# Abrir navegador: http://localhost:8501
```

# Dashboard
Disponible en ```http://localhost:8501``

## Funcionalidades:

- 📈 Visión general del sistema

- 🎯 Predicciones individuales y por lotes

- 📊 Análisis avanzado y segmentación

- ⚙️ Configuración del sistema

# 🤝 Contribución
1. Fork el proyecto

2. Crear rama feature (git checkout -b feature/AmazingFeature)

3. Commit cambios (git commit -m 'Add AmazingFeature')

4. Push a la rama (git push origin feature/AmazingFeature)

5. Abrir Pull Request

# 📄 Licencia
Distribuido bajo MIT License. Ver ```LICENSE``` para más información.


## 🚀 **Comandos para Ejecutar en Producción**

```bash
# 1. Construir y ejecutar con Docker Compose
make docker-up

# 2. Verificar que todo funcione
curl http://localhost:8000/health
# Abrir en navegador: http://localhost:8501

# 3. Probar el flujo completo
# - Dashboard: http://localhost:8501
# - API Docs: http://localhost:8000/docs
# - Predicciones individuales y por lotes
# - Análisis avanzado

# 4. Ejecutar tests dentro de Docker
docker-compose exec api pytest tests/ -v

# 5. Ver logs
make docker-logs

# 6. Entrenar nuevo modelo
make docker-train

# 7. Para desarrollo local (sin Docker)
make install
make generate-data
make train-model
make run-api      # Terminal 1
make run-dashboard # Terminal 2