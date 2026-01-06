"""
Configuration management for Huxley.

Provides a hierarchical configuration system with support for:
- Environment variables
- Configuration files (YAML, TOML)
- Runtime overrides
- Validation and type coercion

Configuration precedence (highest to lowest):
1. Runtime overrides
2. Environment variables
3. Configuration files
4. Default values
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMProviderConfig(BaseSettings):
    """Configuration for a single LLM provider."""

    model_config = SettingsConfigDict(extra="allow")

    api_key: SecretStr | None = None
    base_url: str | None = None
    organization: str | None = None
    timeout: float = 60.0
    max_retries: int = 3


class DatabaseConfig(BaseSettings):
    """Database connection configuration."""

    model_config = SettingsConfigDict(env_prefix="HUXLEY_DB_")

    driver: Literal["sqlite", "postgresql"] = "sqlite"
    host: str = "localhost"
    port: int = 5432
    database: str = "huxley"
    username: str | None = None
    password: SecretStr | None = None
    pool_size: int = 5
    pool_overflow: int = 10

    @property
    def url(self) -> str:
        """Build database connection URL."""
        if self.driver == "sqlite":
            return f"sqlite+aiosqlite:///{self.database}.db"
        elif self.driver == "postgresql":
            auth = ""
            if self.username:
                auth = self.username
                if self.password:
                    auth += f":{self.password.get_secret_value()}"
                auth += "@"
            return f"postgresql+asyncpg://{auth}{self.host}:{self.port}/{self.database}"
        raise ValueError(f"Unsupported database driver: {self.driver}")


class RedisConfig(BaseSettings):
    """Redis connection configuration."""

    model_config = SettingsConfigDict(env_prefix="HUXLEY_REDIS_")

    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: SecretStr | None = None
    ssl: bool = False

    @property
    def url(self) -> str:
        """Build Redis connection URL."""
        scheme = "rediss" if self.ssl else "redis"
        auth = ""
        if self.password:
            auth = f":{self.password.get_secret_value()}@"
        return f"{scheme}://{auth}{self.host}:{self.port}/{self.db}"


class ServerConfig(BaseSettings):
    """API server configuration."""

    model_config = SettingsConfigDict(env_prefix="HUXLEY_SERVER_")

    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    reload: bool = False
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])
    api_key: SecretStr | None = None  # Optional API key for server auth


class LoggingConfig(BaseSettings):
    """Logging configuration."""

    model_config = SettingsConfigDict(env_prefix="HUXLEY_LOG_")

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    format: Literal["json", "console"] = "console"
    file: Path | None = None


class ComputeConfig(BaseSettings):
    """Distributed compute configuration."""

    model_config = SettingsConfigDict(env_prefix="HUXLEY_COMPUTE_")

    backend: Literal["local", "redis", "kubernetes"] = "local"
    max_workers: int = 4
    task_timeout: float = 3600.0  # 1 hour default
    gpu_memory_fraction: float = 0.9


class HuxleyConfig(BaseSettings):
    """
    Root configuration for Huxley.

    All configuration values can be set via environment variables
    with the HUXLEY_ prefix, or via a configuration file.
    """

    model_config = SettingsConfigDict(
        env_prefix="HUXLEY_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    # Environment
    env: Literal["development", "staging", "production"] = "development"
    debug: bool = False

    # Data directories
    data_dir: Path = Field(default_factory=lambda: Path.home() / ".huxley")
    cache_dir: Path | None = None

    # Sub-configurations
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    redis: RedisConfig = Field(default_factory=RedisConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    compute: ComputeConfig = Field(default_factory=ComputeConfig)

    # LLM provider configurations (keyed by provider name)
    providers: dict[str, LLMProviderConfig] = Field(default_factory=dict)

    # Default provider
    default_provider: str = "openai"
    default_model: str = "gpt-4"

    @field_validator("data_dir", "cache_dir", mode="before")
    @classmethod
    def expand_path(cls, v: Any) -> Path | None:
        """Expand ~ and environment variables in paths."""
        if v is None:
            return None
        if isinstance(v, str):
            return Path(v).expanduser()
        return v

    @property
    def effective_cache_dir(self) -> Path:
        """Get the effective cache directory."""
        return self.cache_dir or self.data_dir / "cache"

    def get_provider_config(self, provider: str) -> LLMProviderConfig:
        """Get configuration for a specific provider."""
        if provider in self.providers:
            return self.providers[provider]
        return LLMProviderConfig()

    def ensure_directories(self) -> None:
        """Ensure all required directories exist."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.effective_cache_dir.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_config() -> HuxleyConfig:
    """
    Get the global Huxley configuration.

    This is cached for performance. Call `get_config.cache_clear()`
    to reload configuration.
    """
    return HuxleyConfig()


def configure(**overrides: Any) -> HuxleyConfig:
    """
    Configure Huxley with explicit overrides.

    This clears the cached configuration and creates a new one
    with the provided overrides.
    """
    get_config.cache_clear()
    return HuxleyConfig(**overrides)
