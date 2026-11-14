from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    database_url: str = "postgresql://madde:madde@localhost:5442/madde"
    embedding_dim: int = 1024
    llm_model: str = "claude-opus-5-5"
    anthropic_api_key: str | None = None  # unset: the SDK resolves credentials itself


@lru_cache
def get_settings() -> Settings:
    return Settings()
