from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Database (SQLite by default; override via .env for PostgreSQL)
    database_url: str = "sqlite+aiosqlite:///./resume_platform.db"
    sync_database_url: str = "sqlite:///./resume_platform.db"

    # Models
    spacy_model: str = "en_core_web_lg"
    embedding_model: str = "all-MiniLM-L6-v2"

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    debug: bool = False
    log_level: str = "INFO"

    # ATS Weights
    skill_weight: float = 0.40
    experience_weight: float = 0.25
    education_weight: float = 0.20
    certification_weight: float = 0.10
    language_weight: float = 0.05


@lru_cache
def get_settings() -> Settings:
    return Settings()
