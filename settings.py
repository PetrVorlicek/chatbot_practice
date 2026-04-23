from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    api_url: str
    api_key: str
    semantic_db_path: str = "ai/semantic_docs.sqlite3"
    tickets_db_path: str = "db/tickets.sqlite3"


settings = Settings(_env_file=".env")
