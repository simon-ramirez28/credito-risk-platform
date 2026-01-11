"""
Configuración de la API.
"""
import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()


class Settings(BaseSettings):
    """Configuración de la aplicación."""
    
    # API
    api_host: str = os.getenv("API_HOST", "0.0.0.0")
    api_port: int = int(os.getenv("API_PORT", 8000))
    api_reload: bool = os.getenv("API_RELOAD", "true").lower() == "true"
    api_debug: bool = os.getenv("API_DEBUG", "false").lower() == "true"
    
    # Modelo
    model_path: str = os.getenv("MODEL_PATH", "models/random_forest_model_20260110_211947.pkl")
    model_metadata_path: str = os.getenv(
        "MODEL_METADATA_PATH", 
        "models/random_forest_model_20260110_211947_metadata.json"
    )
    
    # Database
    database_url: str = os.getenv("DATABASE_URL", "sqlite:///./data/creditos.db")
    
    # Security
    secret_key: str = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
    algorithm: str = os.getenv("ALGORITHM", "HS256")
    access_token_expire_minutes: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 30))
    
    # Redis
    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    
    # Paths
    base_dir: Path = Path(__file__).parent.parent.parent
    data_dir: Path = base_dir / "data"
    models_dir: Path = base_dir / "models"
    logs_dir: Path = base_dir / "logs"
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Instancia global de configuración
settings = Settings()

# Crear directorios si no existen
settings.data_dir.mkdir(exist_ok=True)
settings.models_dir.mkdir(exist_ok=True)
settings.logs_dir.mkdir(exist_ok=True)