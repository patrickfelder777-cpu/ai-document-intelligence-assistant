
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_ask_success():
    """The API returns an AI answer with supporting sources."""

    mock_result = {
        "question": "What is machine learning?",
        "answer": "Machine learning identifies patterns [Source 1, Page 2].",
        "document_id": None,
        "total_sources": 1,
        "sources": [
            {
                "text": "Machine learning identifies patterns.",
                "document_id": "test_document",
                "page_number": 2,
                "chunk_id": 0,
                "distance": 0.2,
            }
        ],
    }

    with patch("app.main.answer_question", return_value=mock_result):
        response = client.post(
            "/ask",
            json={
                "question": "What is machine learning?",
                "top_k": 3,
            },
        )

    assert response.status_code == 200
    assert response.json()["answer"] == mock_result["answer"]
    assert response.json()["total_sources"] == 1


def test_ask_empty_question():
    """Empty questions must be rejected."""

    response = client.post(
        "/ask",
        json={"question": ""},
    )

    assert response.status_code == 422


def test_ask_whitespace_question():
    """Whitespace-only questions must be rejected."""

    response = client.post(
        "/ask",
        json={"question": "   "},
    )

    assert response.status_code == 400


def test_ask_invalid_top_k():
    """The API must reject invalid retrieval limits."""

    response = client.post(
        "/ask",
        json={
            "question": "Explain machine learning.",
            "top_k": 25,
        },
    )

    assert response.status_code == 422


def test_ask_document_filter():
    """The selected document ID must reach the RAG engine."""

    mock_result = {
        "question": "Summarize the document.",
        "answer": "This document discusses Python.",
        "document_id": "document_123",
        "total_sources": 0,
        "sources": [],
    }

    with patch(
        "app.main.answer_question",
        return_value=mock_result,
    ) as mock_answer:
        response = client.post(
            "/ask",
            json={
                "question": "Summarize the document.",
                "document_id": "document_123",
                "top_k": 2,
            },
        )

    assert response.status_code == 200

    mock_answer.assert_called_once_with(
        question="Summarize the document.",
        document_id="document_123",
        top_k=2,
    )


def test_ask_llm_failure():
    """LLM service failures must return a controlled error."""

    with patch(
        "app.main.answer_question",
        side_effect=RuntimeError("LLM unavailable"),
    ):
        response = client.post(
            "/ask",
            json={"question": "Explain neural networks."},
        )

    assert response.status_code == 502
    assert response.json()["detail"] == (
        "Unable to generate an AI answer right now."
    )
