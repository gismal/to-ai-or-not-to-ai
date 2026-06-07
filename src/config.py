import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import SecretStr 

class Settings(BaseSettings):
    API_KEY: SecretStr
    MODEL_THRESHOLD: float = 0.75
    GRAY_AREA_MARGIN: float = 0.35
    DEBUG: bool = False
    DATABASE_URL: str 

    model_config = SettingsConfigDict(
        env_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"),
        env_file_encoding= "utf-8"
    )
    
# Singleton Pattern: We'll reach to settings from all over the project with one instance
settings = Settings()