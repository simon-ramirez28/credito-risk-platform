#!/bin/bash
# Health check para servicios

# Verificar API
check_api() {
    curl -f http://localhost:8000/health > /dev/null 2>&1
    return $?
}

# Verificar Dashboard
check_dashboard() {
    curl -f http://localhost:8501/healthz > /dev/null 2>&1
    return $?
}

# Verificar PostgreSQL
check_postgres() {
    pg_isready -h localhost -p 5432 -U credit_user > /dev/null 2>&1
    return $?
}

# Verificar Redis
check_redis() {
    redis-cli -h localhost ping > /dev/null 2>&1
    return $?
}

# Ejecutar checks basados en argumento
case "$1" in
    api)
        check_api
        ;;
    dashboard)
        check_dashboard
        ;;
    postgres)
        check_postgres
        ;;
    redis)
        check_redis
        ;;
    all)
        check_api && check_dashboard && check_postgres && check_redis
        ;;
    *)
        echo "Usage: $0 {api|dashboard|postgres|redis|all}"
        exit 1
        ;;
esac

exit $?