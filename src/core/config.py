from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    environment: str = "development"

    llm_provider: str = "openai"

    openai_api_key: str | None = None
    openai_base_url: str | None = None

    local_llm_base_url: str | None = None
    local_llm_api_key: str | None = None

    log_level: str = "INFO"

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )


settings = Settings()
