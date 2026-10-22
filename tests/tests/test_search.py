
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.vector_store import VectorStore


client = TestClient(app)


@pytest.fixture
def test_store(tmp_path, monkeypatch):
    """Use a temporary ChromaDB database for each test."""
    store = VectorStore(
        persist_directory=str(tmp_path / "search_chroma")
    )

    monkeypatch.setattr(
        "app.main.VectorStore",
        lambda: store,
    )

    return store


def test_search_empty_database(test_store):
    response = client.post(
        "/search",
        json={"query": "What is machine learning?"},
    )

    assert response.status_code == 200
    assert response.json()["total_results"] == 0
    assert response.json()["results"] == []


def test_search_returns_relevant_results(test_store):
    chunks = [
        {
            "chunk_id": 0,
            "page_number": 1,
            "text": "Machine learning identifies patterns in data.",
        },
        {
            "chunk_id": 1,
            "page_number": 2,
            "text": "Docker packages applications into containers.",
        },
    ]

    test_store.add_chunks(chunks, document_id="test_document")

    response = client.post(
        "/search",
        json={
            "query": "How does machine learning find patterns?",
            "top_k": 2,
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["total_results"] == 2
    assert "Machine learning" in data["results"][0]["text"]


def test_search_respects_top_k(test_store):
    chunks = [
        {
            "chunk_id": i,
            "page_number": 1,
            "text": f"Machine learning topic number {i}.",
        }
        for i in range(5)
    ]

    test_store.add_chunks(chunks, document_id="test_document")

    response = client.post(
        "/search",
        json={"query": "machine learning", "top_k": 2},
    )

    assert response.status_code == 200
    assert response.json()["total_results"] == 2


def test_search_rejects_empty_query(test_store):
    response = client.post(
        "/search",
        json={"query": ""},
    )

    assert response.status_code == 422


def test_search_rejects_whitespace_query(test_store):
    response = client.post(
        "/search",
        json={"query": "   "},
    )

    assert response.status_code == 400


def test_search_rejects_invalid_top_k(test_store):
    response = client.post(
        "/search",
        json={"query": "machine learning", "top_k": 0},
    )

    assert response.status_code == 422


def test_search_preserves_metadata(test_store):
    test_store.add_chunks(
        [
            {
                "chunk_id": 7,
                "page_number": 3,
                "text": "Vector databases store numerical embeddings.",
            }
        ],
        document_id="document_123",
    )

    response = client.post(
        "/search",
        json={"query": "Where are embeddings stored?"},
    )

    assert response.status_code == 200

    result = response.json()["results"][0]

    assert result["document_id"] == "document_123"
    assert result["page_number"] == 3
    assert result["chunk_id"] == 7
    assert isinstance(result["distance"], float)


def test_search_filters_by_document_id(test_store):
    """Search results must belong to the requested document."""

    test_store.add_chunks(
        [{
            "chunk_id": 0,
            "page_number": 1,
            "text": "Machine learning identifies patterns in data.",
        }],
        document_id="document_A",
    )

    test_store.add_chunks(
        [{
            "chunk_id": 0,
            "page_number": 2,
            "text": "Docker containers package applications.",
        }],
        document_id="document_B",
    )

    response = client.post(
        "/search",
        json={
            "query": "What is machine learning?",
            "top_k": 3,
            "document_id": "document_A",
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["total_results"] == 1
    assert data["results"][0]["document_id"] == "document_A"
    assert "Machine learning" in data["results"][0]["text"]


def test_search_nonexistent_document(test_store):
    """Unknown document IDs must return no results."""

    test_store.add_chunks(
        [{
            "chunk_id": 0,
            "page_number": 1,
            "text": "Artificial intelligence processes information.",
        }],
        document_id="existing_document",
    )

    response = client.post(
        "/search",
        json={
            "query": "Artificial intelligence",
            "document_id": "nonexistent_document",
        },
    )

    assert response.status_code == 200
    assert response.json()["total_results"] == 0
    assert response.json()["results"] == []


def test_search_without_document_filter(test_store):
    """Search all documents when no ID is provided."""

    test_store.add_chunks(
        [{
            "chunk_id": 0,
            "page_number": 1,
            "text": "Python supports machine learning applications.",
        }],
        document_id="document_A",
    )

    test_store.add_chunks(
        [{
            "chunk_id": 0,
            "page_number": 1,
            "text": "FastAPI provides tools for building APIs.",
        }],
        document_id="document_B",
    )

    response = client.post(
        "/search",
        json={
            "query": "Python applications",
            "top_k": 5,
            "document_id": None,
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["total_results"] == 2

    document_ids = {
        result["document_id"]
        for result in data["results"]
    }

    assert document_ids == {"document_A", "document_B"}


def test_search_filter_preserves_page_numbers(test_store):
    """Filtered searches must retain page references."""

    test_store.add_chunks(
        [{
            "chunk_id": 5,
            "page_number": 7,
            "text": "Neural networks learn complex representations.",
        }],
        document_id="neural_network_document",
    )

    response = client.post(
        "/search",
        json={
            "query": "How do neural networks learn?",
            "document_id": "neural_network_document",
        },
    )

    assert response.status_code == 200

    result = response.json()["results"][0]

    assert result["document_id"] == "neural_network_document"
    assert result["page_number"] == 7
    assert result["chunk_id"] == 5
