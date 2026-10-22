
import pytest

from app.vector_store import VectorStore


@pytest.fixture
def vector_store(tmp_path):
    """Create an isolated ChromaDB database for each test."""
    return VectorStore(
        persist_directory=str(tmp_path / "test_chroma")
    )


@pytest.fixture
def sample_chunks():
    return [
        {
            "chunk_id": 0,
            "page_number": 1,
            "text": (
                "Machine learning algorithms identify "
                "patterns in data."
            ),
        },
        {
            "chunk_id": 1,
            "page_number": 2,
            "text": (
                "Vector databases store embeddings "
                "for semantic similarity search."
            ),
        },
        {
            "chunk_id": 2,
            "page_number": 3,
            "text": (
                "Docker containers package applications "
                "for deployment."
            ),
        },
    ]


def test_empty_database(vector_store):
    assert vector_store.collection.count() == 0
    assert vector_store.search("machine learning") == []


def test_add_chunks(vector_store, sample_chunks):
    added = vector_store.add_chunks(
        sample_chunks,
        document_id="test_document",
    )

    assert added == 3
    assert vector_store.collection.count() == 3


def test_add_empty_chunks(vector_store):
    added = vector_store.add_chunks(
        [],
        document_id="empty_document",
    )

    assert added == 0
    assert vector_store.collection.count() == 0


def test_search_returns_results(vector_store, sample_chunks):
    vector_store.add_chunks(
        sample_chunks,
        document_id="test_document",
    )

    results = vector_store.search(
        "How do vector databases store embeddings?",
        top_k=3,
    )

    assert len(results) == 3

    assert "text" in results[0]
    assert "document_id" in results[0]
    assert "page_number" in results[0]
    assert "chunk_id" in results[0]
    assert "distance" in results[0]


def test_search_returns_relevant_chunk(vector_store, sample_chunks):
    vector_store.add_chunks(
        sample_chunks,
        document_id="test_document",
    )

    results = vector_store.search(
        "How are embeddings stored for similarity search?",
        top_k=3,
    )

    assert "Vector databases" in results[0]["text"]
    assert results[0]["page_number"] == 2


def test_search_respects_top_k(vector_store, sample_chunks):
    vector_store.add_chunks(
        sample_chunks,
        document_id="test_document",
    )

    results = vector_store.search(
        "machine learning",
        top_k=2,
    )

    assert len(results) == 2


def test_search_top_k_larger_than_collection(
    vector_store,
    sample_chunks,
):
    vector_store.add_chunks(
        sample_chunks,
        document_id="test_document",
    )

    results = vector_store.search(
        "machine learning",
        top_k=10,
    )

    assert len(results) == 3


def test_invalid_top_k(vector_store):
    with pytest.raises(ValueError, match="top_k"):
        vector_store.search("machine learning", top_k=0)


def test_metadata_preserved(vector_store, sample_chunks):
    vector_store.add_chunks(
        sample_chunks,
        document_id="test_document",
    )

    results = vector_store.search(
        "Docker deployment",
        top_k=3,
    )

    docker_result = next(
        result
        for result in results
        if "Docker containers" in result["text"]
    )

    assert docker_result["document_id"] == "test_document"
    assert docker_result["page_number"] == 3
    assert docker_result["chunk_id"] == 2


def test_database_persistence(tmp_path, sample_chunks):
    database_path = str(tmp_path / "persistent_chroma")

    first_store = VectorStore(
        persist_directory=database_path
    )

    first_store.add_chunks(
        sample_chunks,
        document_id="persistent_document",
    )

    second_store = VectorStore(
        persist_directory=database_path
    )

    assert second_store.collection.count() == 3
