from functools import lru_cache

from dotenv import load_dotenv
from pydantic_settings import BaseSettings

# Force .env values to win over any stale Windows system environment variables
load_dotenv(override=True)


class Settings(BaseSettings):
    twilio_account_sid: str = ""
    twilio_auth_token: str = ""
    twilio_phone_number: str = ""

    deepgram_api_key: str = ""
    openai_api_key: str = ""

    supabase_url: str = ""
    supabase_service_key: str = ""

    orchestrator_base_url: str = "http://localhost:8080"

    log_level: str = "INFO"
    environment: str = "development"

    model_config = {"env_file": ".env", "case_sensitive": False, "extra": "ignore"}


@lru_cache
def get_settings() -> Settings:
    return Settings()
