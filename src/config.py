import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import SecretStr, model_validator


class Settings(BaseSettings):
    API_KEY: SecretStr
    MODEL_THRESHOLD: float = 0.75
    GRAY_AREA_MARGIN: float = 0.35
    DEBUG: bool = False
    DATABASE_URL: str
    REDIS_URL: str = "redis://localhost:6379/0"
    DRIFT_THRESHOLD: float = 0.15
    ALLOWED_ORIGINS: list[str] = [
        "http://localhost:8000",
        "http://localhost:3000",
        "chrome-extension://*",
    ]

    model_config = SettingsConfigDict(
        env_file=os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @model_validator(mode="after")
    def validate_threshold(self) -> "Settings":
        if self.GRAY_AREA_MARGIN >= self.MODEL_THRESHOLD:
            raise ValueError("GRAY_AREA_MARGIN must be smaller than MODEL_THRESHOLD")


# Singleton Pattern: We'll reach to settings from all over the project with one instance
settings = Settings()
