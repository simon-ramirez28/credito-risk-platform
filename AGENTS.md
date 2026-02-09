# AGENTS.md - Guía para Agentes de Código

## Comandos de Build/Lint/Test

### Testing
```bash
# Todos los tests con cobertura
make test
pytest tests/ -v --cov=src --cov-report=html --cov-report=term-missing

# Un solo test específico
pytest tests/test_models.py -v
pytest tests/test_models.py::test_train_and_evaluate -v

# Tests con salida detallada
pytest tests/ -v --tb=short
```

### Linting y Formato
```bash
# Verificar calidad de código
make lint
flake8 src/ --count --max-complexity=10 --max-line-length=127 --statistics
mypy src/ --ignore-missing-imports
black --check src/

# Formatear código automáticamente
make format
black src/
isort src/
```

### Ejecución Local
```bash
# Instalar dependencias
make install

# Pipeline completo
python main.py todo --clientes 1000

# API local
make run-api

# Dashboard
make run-dashboard

# Docker
make docker-up
make docker-down
```

## Guías de Estilo de Código

### Imports
- Orden: stdlib → terceros → locales
- Separar con línea en blanco entre grupos
- Usar imports absolutos, no relativos
- Evitar imports con `*`

```python
import logging
from pathlib import Path
from typing import Dict, Any

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from src.models.train_model import CreditRiskModel
```

### Formato
- Black como formateador (line-length: 127)
- isort para ordenar imports
- Usar comillas dobles para strings
- Línea máxima: 127 caracteres

### Naming Conventions
- **Clases**: PascalCase (`CreditRiskModel`, `FeatureEngineer`)
- **Funciones/Variables**: snake_case (`prepare_data`, `feature_names`)
- **Constantes**: UPPER_SNAKE_CASE
- **Módulos**: snake_case.py
- **Privados**: prefijo `_` (`_initialize_model`)

### Type Hints
- Usar en todas las funciones públicas
- Importar desde `typing`: `Dict`, `List`, `Optional`, `Tuple`, `Any`
- Tipos de retorno obligatorios en funciones públicas

```python
def prepare_data(self, df: pd.DataFrame, test_size: float = 0.2) -> Tuple:
```

### Docstrings
- Estilo Google/NumPy
- Documentar Args, Returns, Raises
- Descripción concisa en primera línea

```python
def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Any]:
    """Entrena el modelo.
    
    Args:
        X_train: Features de entrenamiento
        y_train: Target de entrenamiento
    
    Returns:
        Diccionario con métricas
    """
```

### Manejo de Errores
- Usar excepciones específicas, no genéricas
- Validar inputs temprano con mensajes claros
- Usar logging en lugar de print
- No capturar excepciones silenciosamente

```python
if self.target_name not in df.columns:
    raise ValueError(f"Target column '{self.target_name}' no encontrada")

logger.error("[X]  No se encontraron archivos procesados")
```

### Logging
- Usar módulo logging configurado
- Niveles: INFO para progreso, ERROR para fallos
- Formato estándar con timestamps
- Usar emojis/spacers para visibilidad en CLI

### Tests
- Usar pytest, no unittest
- Fixtures con `tmp_path` para archivos temporales
- Nombres descriptivos: `test_` + descripción
- Datos de prueba con `make_classification`

### Estructura del Proyecto
- `src/`: Código fuente organizado por módulos
- `tests/`: Tests unitarios e integración
- `main.py`: Punto de entrada CLI
- `Makefile`: Comandos automatizados

### Comentarios
- Evitar comentarios obvios
- Explicar el "por qué", no el "qué"
- Usar `#` para notas temporales (TODO, FIXME)

### Commits
- Mensajes descriptivos en inglés o español consistente
- Estructura: tipo(scope): descripción
- Tipos: feat, fix, docs, style, refactor, test, chore
