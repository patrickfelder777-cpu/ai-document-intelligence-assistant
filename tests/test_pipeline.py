from unittest.mock import patch

import pymupdf
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.vector_store import VectorStore



@pytest.fixture(autouse=True)
def isolate_vector_store(tmp_path, monkeypatch):
    """Redirect API database operations to a temporary database."""

    real_vector_store = VectorStore

    def create_test_store():
        return real_vector_store(
            persist_directory=str(tmp_path / "isolated_chroma")
        )

    monkeypatch.setattr(
        "app.main.VectorStore",
        create_test_store,
    )

from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def create_test_pdf(page_texts: list[str]) -> bytes:
    """Create a PDF containing the provided page texts."""
    document = pymupdf.open()

    for text in page_texts:
        page = document.new_page()
        page.insert_textbox(
            pymupdf.Rect(50, 50, 550, 750),
            text,
            fontsize=10,
        )

    pdf_bytes = document.tobytes()
    document.close()

    return pdf_bytes


def upload_test_pdf(pdf_bytes: bytes):
    """Upload a PDF through the FastAPI endpoint."""
    return client.post(
        "/upload",
        files={
            "file": (
                "integration_test.pdf",
                pdf_bytes,
                "application/pdf",
            )
        },
    )


def test_upload_returns_chunks():
    pdf_bytes = create_test_pdf([
        "Machine learning helps computers identify patterns."
    ])

    response = upload_test_pdf(pdf_bytes)

    assert response.status_code == 200

    data = response.json()

    assert "chunks" in data
    assert "total_chunks" in data
    assert data["total_chunks"] == 1
    assert len(data["chunks"]) == 1

    assert "Machine learning" in data["chunks"][0]["text"]


def test_pipeline_preserves_page_numbers():
    pdf_bytes = create_test_pdf([
        "First page contains machine learning information.",
        "Second page contains retrieval augmented generation.",
    ])

    response = upload_test_pdf(pdf_bytes)

    assert response.status_code == 200

    data = response.json()

    assert data["pages_extracted"] == 2
    assert data["total_chunks"] == 2

    assert data["chunks"][0]["page_number"] == 1
    assert data["chunks"][1]["page_number"] == 2


def test_pipeline_generates_unique_chunk_ids():
    pdf_bytes = create_test_pdf([
        "Artificial intelligence and document processing.",
        "Vector databases support semantic search.",
        "Language models generate answers from context.",
    ])

    response = upload_test_pdf(pdf_bytes)

    assert response.status_code == 200

    data = response.json()

    chunk_ids = [
        chunk["chunk_id"]
        for chunk in data["chunks"]
    ]

    assert len(chunk_ids) == len(set(chunk_ids))
    assert chunk_ids == list(range(len(chunk_ids)))


def test_pipeline_handles_blank_page():
    pdf_bytes = create_test_pdf([
        "This page contains useful information.",
        "",
    ])

    response = upload_test_pdf(pdf_bytes)

    assert response.status_code == 200

    data = response.json()

    assert data["pages_extracted"] == 1
    assert data["total_chunks"] == 1
    assert data["chunks"][0]["page_number"] == 1


def test_pipeline_preserves_extracted_text():
    original_text = (
        "Retrieval augmented generation combines "
        "document retrieval with language models."
    )

    pdf_bytes = create_test_pdf([original_text])

    response = upload_test_pdf(pdf_bytes)

    assert response.status_code == 200

    data = response.json()

    assert original_text in data["pages"][0]["text"]
    assert original_text in data["chunks"][0]["text"]

    assert data["total_characters"] == sum(
        len(page["text"]) for page in data["pages"]
    )


def test_upload_stores_chunks_in_chromadb(tmp_path):
    """Verify uploaded PDF chunks are saved in ChromaDB."""

    test_store = VectorStore(
        persist_directory=str(tmp_path / "test_chroma")
    )

    pdf_bytes = create_test_pdf([
        "Machine learning identifies patterns in data.",
        "Vector databases support semantic search.",
    ])

    with patch("app.main.VectorStore", return_value=test_store):
        response = upload_test_pdf(pdf_bytes)

    assert response.status_code == 200

    data = response.json()

    assert "document_id" in data
    assert data["total_chunks"] == 2
    assert data["stored_chunks"] == 2

    assert test_store.collection.count() == 2

    stored_data = test_store.collection.get(
        include=["documents", "metadatas"]
    )

    assert len(stored_data["documents"]) == 2

    assert all(
        metadata["document_id"] == data["document_id"]
        for metadata in stored_data["metadatas"]
    )
