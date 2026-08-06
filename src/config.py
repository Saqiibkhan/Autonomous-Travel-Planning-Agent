"""
Central application configuration.

Every setting the agent needs (API keys, model name, timeouts, retry counts)
lives here and is loaded once from environment variables / a .env file.
No other module should call os.getenv() directly -- they should import
`settings` from this file instead. That gives us one source of truth and
makes testing easy (we can monkeypatch `settings` in tests instead of
mucking with real env vars).
"""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Pydantic Settings reads from a .env file first, then real environment
    # variables (which always win) -- and it validates + type-casts everything.
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- App metadata ---
    app_name: str = "Autonomous Travel Planning Agent"
    app_env: str = Field(default="development")  # development | production
    log_level: str = Field(default="INFO")

    # --- LLM (OpenRouter / OpenAI-compatible) ---
    llm_api_key: str = Field(default="")
    llm_base_url: str = Field(default="https://openrouter.ai/api/v1")
    llm_model: str = Field(default="openai/gpt-oss-20b:free")

    # --- External tool API keys (filled in as each tool is built) ---
    search_api_key: str = Field(default="")
    places_api_key: str = Field(default="")
    weather_api_key: str = Field(default="")
    routing_api_key: str = Field(default="")
    currency_api_key: str = Field(default="")

    # --- HTTP / retry behaviour, shared by every tool ---
    request_timeout_seconds: float = Field(default=15.0)
    max_tool_retries: int = Field(default=3)
    max_reflection_retries: int = Field(default=3)


@lru_cache
def get_settings() -> Settings:
    """
    Cached settings loader.

    lru_cache means the .env file / environment is only read once per
    process, and every part of the app that calls get_settings() gets back
    the exact same Settings object.
    """
    return Settings()


# Convenience singleton -- most modules will just do:
#   from src.config import settings
settings = get_settings()
