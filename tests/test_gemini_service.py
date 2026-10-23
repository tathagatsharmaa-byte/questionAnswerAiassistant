"""Tests for Gemini answer generation without network calls."""

from unittest.mock import Mock, patch

import pytest
from google.genai import errors

from app.config.settings import Settings
from app.services.gemini_service import GeminiService, GeminiServiceError


def make_settings() -> Settings:
    return Settings(
        gemini_api_key="test-key",
        model="test-model",
        temperature=0.4,
    )


@patch("app.services.gemini_service.genai.Client")
def test_initializes_client_with_configured_key(gemini_client: Mock) -> None:
    GeminiService(make_settings())

    gemini_client.assert_called_once_with(api_key="test-key")


def test_generates_answer_with_conversation_history() -> None:
    client = Mock()
    client.models.generate_content.return_value.text = "A concise answer."
    service = GeminiService(make_settings(), client=client)
    messages = [
        {"role": "user", "content": "What is a unit test?"},
        {"role": "assistant", "content": "A test of a small unit of code."},
        {"role": "user", "content": "Why use one?"},
    ]

    answer = service.generate_answer(messages)

    assert answer == "A concise answer."
    call = client.models.generate_content.call_args.kwargs
    assert call["model"] == "test-model"
    assert call["config"].temperature == 0.4
    assert call["config"].system_instruction
    assert call["contents"] == [
        {"role": "user", "parts": [{"text": "What is a unit test?"}]},
        {"role": "model", "parts": [{"text": "A test of a small unit of code."}]},
        {"role": "user", "parts": [{"text": "Why use one?"}]},
    ]


def test_rejects_empty_api_response() -> None:
    client = Mock()
    client.models.generate_content.return_value.text = "  "
    service = GeminiService(make_settings(), client=client)

    with pytest.raises(GeminiServiceError, match="empty answer"):
        service.generate_answer([{"role": "user", "content": "Hello"}])


def test_api_error_is_logged_without_provider_details(
    caplog: pytest.LogCaptureFixture,
) -> None:
    client = Mock()
    client.models.generate_content.side_effect = errors.APIError(
        code=500,
        response_json={"error": {"message": "sensitive provider details"}},
    )
    service = GeminiService(make_settings(), client=client)

    with pytest.raises(GeminiServiceError, match="could not complete"):
        service.generate_answer([{"role": "user", "content": "Hello"}])

    assert "sensitive provider details" not in caplog.text
    assert "APIError" in caplog.text


def test_access_error_gives_key_guidance_without_secret(
    caplog: pytest.LogCaptureFixture,
) -> None:
    secret = "rejected-test-key"
    client = Mock()
    client.models.generate_content.side_effect = errors.APIError(
        code=403,
        response_json={"error": {"message": secret}},
    )
    service = GeminiService(
        Settings(
            gemini_api_key=secret,
            model="test-model",
            temperature=0.4,
        ),
        client=client,
    )

    with pytest.raises(GeminiServiceError, match="Check GEMINI_API_KEY"):
        service.generate_answer([{"role": "user", "content": "Hello"}])

    assert secret not in caplog.text
