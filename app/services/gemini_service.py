"""Gemini-backed answer generation."""

import logging
from collections.abc import Sequence
from typing import Literal, TypedDict

from google import genai
from google.genai import errors, types

from app.config.settings import Settings
from app.prompts.qa_prompt import SYSTEM_PROMPT

logger = logging.getLogger(__name__)


class ConversationMessage(TypedDict):
    """A user or assistant message in the conversation."""

    role: Literal["user", "assistant"]
    content: str


class GeminiServiceError(RuntimeError):
    """A safe, user-facing error from answer generation."""


class GeminiService:
    """Encapsulate Gemini client setup and answer generation."""

    def __init__(
        self,
        settings: Settings,
        client: genai.Client | None = None,
    ) -> None:
        self._settings = settings
        self._client = (
            client
            if client is not None
            else genai.Client(api_key=settings.gemini_api_key)
        )

    def generate_answer(
        self,
        messages: Sequence[ConversationMessage],
    ) -> str:
        """Generate an answer using the system prompt and conversation history."""
        contents = [
            {
                "role": "model" if message["role"] == "assistant" else "user",
                "parts": [{"text": message["content"]}],
            }
            for message in messages
        ]

        try:
            response = self._client.models.generate_content(
                model=self._settings.model,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=self._settings.temperature,
                ),
            )
        except errors.APIError as error:
            status_code = getattr(error, "code", None)
            logger.error(
                "Gemini API request failed (%s, status=%s).",
                type(error).__name__,
                status_code,
            )
            if status_code in (400, 401, 403):
                raise GeminiServiceError(
                    "Gemini rejected the API key or denied access. Check "
                    "GEMINI_API_KEY in your .env file and verify that the key "
                    "is enabled for the Gemini API."
                ) from None
            if status_code == 429:
                raise GeminiServiceError(
                    "The Gemini API rate limit or usage quota was reached. "
                    "Check your account's usage and try again later."
                ) from None
            raise GeminiServiceError(
                "The Gemini service could not complete your request. "
                "Please try again."
            ) from None

        answer = response.text
        if not isinstance(answer, str) or not answer.strip():
            logger.error("Gemini API returned an empty answer.")
            raise GeminiServiceError(
                "The Gemini service returned an empty answer. Please try again."
            )

        logger.info("Generated an answer successfully.")
        return answer
