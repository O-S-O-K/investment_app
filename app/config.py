from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Investment App"
    environment: str = "dev"
    database_url: str = "sqlite:///./investment_app.db"
    risk_free_rate: float = 0.04
    annualization: int = 252
    max_single_weight: float = 0.35
    min_single_weight: float = 0.0
    target_volatility: float = 0.13
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
