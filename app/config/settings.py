from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Telegram
    bot_token: str
    allowed_user_ids: str = ""
    admin_user_id: int = 0

    # OpenAI
    openai_api_key: str
    openai_base_url: str = "https://api.openai.com/v1"
    llm_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    whisper_model: str = "whisper-1"

    # PostgreSQL
    postgres_host: str = "postgres"
    postgres_port: int = 5432
    postgres_db: str = "second_brain"
    postgres_user: str = "brain"
    postgres_password: str = "change_me_in_production"

    # Qdrant
    qdrant_host: str = "qdrant"
    qdrant_port: int = 6333
    qdrant_collection: str = "notes"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    @property
    def postgres_dsn(self) -> str:
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def initial_allowed_ids(self) -> set[int]:
        if not self.allowed_user_ids.strip():
            return set()
        return {int(uid.strip()) for uid in self.allowed_user_ids.split(",")}


settings = Settings()  # type: ignore[call-arg]
