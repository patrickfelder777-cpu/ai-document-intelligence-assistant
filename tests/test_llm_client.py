
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import httpx
import pytest

from openai import (
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    RateLimitError,
    InternalServerError,
)

from app.llm_client import generate_answer


QUESTION = "What is machine learning?"
CONTEXT = "[Source 1]\nPage: 2\nContent: Machine learning identifies patterns."


@pytest.fixture
def mock_client():
    """Mock the Nebius client without making real API requests."""

    with patch("app.llm_client.get_llm_client") as mock_factory:
        client = MagicMock()
        mock_factory.return_value = client

        with patch.dict(
            "app.llm_client.os.environ",
            {"NEBIUS_MODEL": "test-model"},
        ):
            yield client


def test_generate_answer_success(mock_client):
    """A successful API response returns the generated answer."""

    mock_client.chat.completions.create.return_value = (
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="Machine learning identifies patterns."
                    )
                )
            ]
        )
    )

    result = generate_answer(QUESTION, CONTEXT)

    assert result == "Machine learning identifies patterns."
    mock_client.chat.completions.create.assert_called_once()


def test_generate_answer_empty_question():
    """Empty questions are rejected before calling the API."""

    with pytest.raises(ValueError, match="Question cannot be empty"):
        generate_answer("   ", CONTEXT)


def test_generate_answer_empty_context():
    """Missing document context returns a safe fallback."""

    result = generate_answer(QUESTION, "")

    assert "couldn't find relevant information" in result


def test_generate_answer_authentication_error(mock_client):
    """Invalid API credentials produce a controlled error."""

    response = httpx.Response(
        401,
        request=httpx.Request("POST", "https://example.com/v1/chat"),
    )

    mock_client.chat.completions.create.side_effect = (
        AuthenticationError(
            "Invalid API key",
            response=response,
            body=None,
        )
    )

    with pytest.raises(RuntimeError, match="authentication failed"):
        generate_answer(QUESTION, CONTEXT)


def test_generate_answer_rate_limit(mock_client):
    """Rate limiting produces a controlled error."""

    response = httpx.Response(
        429,
        request=httpx.Request("POST", "https://example.com/v1/chat"),
    )

    mock_client.chat.completions.create.side_effect = (
        RateLimitError(
            "Rate limit exceeded",
            response=response,
            body=None,
        )
    )

    with pytest.raises(RuntimeError, match="rate limit exceeded"):
        generate_answer(QUESTION, CONTEXT)


def test_generate_answer_timeout(mock_client):
    """Timeouts produce a controlled error."""

    request = httpx.Request(
        "POST",
        "https://example.com/v1/chat",
    )

    mock_client.chat.completions.create.side_effect = (
        APITimeoutError(request=request)
    )

    with pytest.raises(RuntimeError, match="timed out"):
        generate_answer(QUESTION, CONTEXT)


def test_generate_answer_connection_error(mock_client):
    """Network failures produce a controlled error."""

    request = httpx.Request(
        "POST",
        "https://example.com/v1/chat",
    )

    mock_client.chat.completions.create.side_effect = (
        APIConnectionError(request=request)
    )

    with pytest.raises(RuntimeError, match="Unable to connect"):
        generate_answer(QUESTION, CONTEXT)


def test_generate_answer_server_error(mock_client):
    """Server errors produce a controlled error."""

    response = httpx.Response(
        500,
        request=httpx.Request("POST", "https://example.com/v1/chat"),
    )

    mock_client.chat.completions.create.side_effect = (
        InternalServerError(
            "Internal server error",
            response=response,
            body=None,
        )
    )

    with pytest.raises(RuntimeError, match="HTTP 500"):
        generate_answer(QUESTION, CONTEXT)


def test_generate_answer_empty_llm_response(mock_client):
    """An empty model response is rejected."""

    mock_client.chat.completions.create.return_value = (
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=None)
                )
            ]
        )
    )

    with pytest.raises(ValueError, match="empty answer"):
        generate_answer(QUESTION, CONTEXT)
