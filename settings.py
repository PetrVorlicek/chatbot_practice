from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    api_url: str
    api_key: str


settings = Settings(_env_file=".env")
