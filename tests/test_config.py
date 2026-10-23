"""Tests for environment-backed settings."""

import pytest

from app.config.settings import ConfigurationError, get_settings


@pytest.fixture(autouse=True)
def prevent_loading_local_dotenv(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.config.settings.load_dotenv", lambda: None)


def test_loads_settings_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_MODEL", "test-model")
    monkeypatch.setenv("GEMINI_TEMPERATURE", "0.7")

    settings = get_settings()

    assert settings.gemini_api_key == "test-key"
    assert settings.model == "test-model"
    assert settings.temperature == 0.7


def test_uses_defaults_for_optional_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    monkeypatch.delenv("GEMINI_TEMPERATURE", raising=False)

    settings = get_settings()

    assert settings.model == "gemini-2.5-flash"
    assert settings.temperature == 0.2


def test_missing_api_key_fails_with_env_setup_guidance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    with pytest.raises(ConfigurationError, match=r"GEMINI_API_KEY.*\.env file"):
        get_settings()


@pytest.mark.parametrize("value", ["not-a-number", "-0.1", "2.1"])
def test_invalid_temperature_fails(
    monkeypatch: pytest.MonkeyPatch,
    value: str,
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_TEMPERATURE", value)

    with pytest.raises(ConfigurationError, match="TEMPERATURE"):
        get_settings()
