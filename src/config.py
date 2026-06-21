import os
from pydantic import SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Auth
    API_KEY: SecretStr

    # Model
    MODEL_THRESHOLD: float = 0.75
    GRAY_AREA_MARGIN: float = 0.35
    DRIFT_THRESHOLD: float = 0.15

    # Infrastructure
    DATABASE_URL: str
    REDIS_URL: str = "redis://redis:6379/0"
    DEBUG: bool = False

    # Security
    GRAFANA_PASSWORD: SecretStr = SecretStr("admin")
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
    def validate_thresholds(self) -> "Settings":
        if self.GRAY_AREA_MARGIN >= self.MODEL_THRESHOLD:
            raise ValueError(
                f"GRAY_AREA_MARGIN ({self.GRAY_AREA_MARGIN}) must be "
                f"smaller than MODEL_THRESHOLD ({self.MODEL_THRESHOLD})"
            )
        return self


settings = Settings()  # type: ignore[call-arg]
