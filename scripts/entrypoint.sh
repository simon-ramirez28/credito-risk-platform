#!/bin/bash
# Script de entrada para contenedores

set -e

echo "🚀 Iniciando aplicación..."

# Esperar a servicios dependientes si es necesario
if [ -n "$WAIT_FOR_HOSTS" ]; then
    echo "⏳ Esperando servicios: $WAIT_FOR_HOSTS"
    for host in $(echo $WAIT_FOR_HOSTS | tr ',' ' '); do
        ./scripts/wait-for-it.sh $host --timeout=60
    done
fi

# Ejecutar migraciones de base de datos si es necesario
if [ "$RUN_MIGRATIONS" = "true" ]; then
    echo "📊 Ejecutando migraciones..."
    python -m alembic upgrade head
fi

# Inicializar datos si es necesario
if [ "$INIT_DATA" = "true" ]; then
    echo "📝 Inicializando datos..."
    python scripts/init_data.py
fi

# Ejecutar el comando principal
exec "$@"