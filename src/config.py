import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    API_KEY: SecretStr
    MODEL_THRESHOLD: float = 0.75
    GRAY_AREA_MARGIN = 0.35
    DEBUG: bool
    
    model_config = SettingsConfigDict(
        env_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"),
        env_file_encoding= "utf-8"
    )
    
    DATABASE_URL: str = PostgresDsn
# Singleton Pattern: We'll reach to settings from all over the project with one instance
settings = Settings()