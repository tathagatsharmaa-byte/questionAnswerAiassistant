"""Environment-backed application settings."""

from dataclasses import dataclass, field
import os

from dotenv import load_dotenv


class ConfigurationError(ValueError):
    """Raised when required application configuration is unavailable."""


@dataclass(frozen=True)
class Settings:
    """Runtime settings loaded from environment variables."""

    gemini_api_key: str = field(repr=False)
    model: str
    temperature: float


def get_settings() -> Settings:
    """Load and validate settings from the environment and local .env file."""
    load_dotenv()

    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise ConfigurationError(
            "GEMINI_API_KEY is missing. Create a .env file from .env.example "
            "and add your Gemini API key."
        )

    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()
    if not model:
        raise ConfigurationError("GEMINI_MODEL must not be empty.")

    raw_temperature = os.getenv("GEMINI_TEMPERATURE", "0.2").strip()
    try:
        temperature = float(raw_temperature)
    except ValueError as error:
        raise ConfigurationError(
            "GEMINI_TEMPERATURE must be a number between 0 and 2."
        ) from error

    if not 0 <= temperature <= 2:
        raise ConfigurationError("GEMINI_TEMPERATURE must be between 0 and 2.")

    return Settings(
        gemini_api_key=api_key,
        model=model,
        temperature=temperature,
    )
