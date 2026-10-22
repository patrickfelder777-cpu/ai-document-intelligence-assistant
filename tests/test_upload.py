import pytest
import pymupdf
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


def create_sample_pdf() -> bytes:
    """Generate a small PDF in memory for testing."""
    document = pymupdf.open()

    page = document.new_page()
    page.insert_text(
        (72, 72),
        "Artificial intelligence can analyze documents."
    )

    pdf_bytes = document.tobytes()
    document.close()

    return pdf_bytes


def test_upload_valid_pdf():
    pdf_bytes = create_sample_pdf()

    response = client.post(
        "/upload",
        files={
            "file": (
                "sample.pdf",
                pdf_bytes,
                "application/pdf",
            )
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["filename"] == "sample.pdf"
    assert data["pages_extracted"] == 1
    assert data["total_characters"] > 0
    assert len(data["pages"]) == 1
    assert data["pages"][0]["page_number"] == 1
    assert "Artificial intelligence" in data["pages"][0]["text"]


def test_upload_rejects_non_pdf_extension():
    response = client.post(
        "/upload",
        files={
            "file": (
                "notes.txt",
                b"This is a text file.",
                "text/plain",
            )
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Only PDF files are supported."


def test_upload_rejects_invalid_content_type():
    response = client.post(
        "/upload",
        files={
            "file": (
                "fake.pdf",
                b"Not a real PDF",
                "text/plain",
            )
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid PDF content type."


def test_upload_rejects_invalid_pdf_content():
    response = client.post(
        "/upload",
        files={
            "file": (
                "fake.pdf",
                b"Not a real PDF",
                "application/pdf",
            )
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "The uploaded file is not a valid PDF."


def test_upload_rejects_oversized_pdf():
    oversized_content = b"%PDF-" + b"x" * (10 * 1024 * 1024)

    response = client.post(
        "/upload",
        files={
            "file": (
                "large.pdf",
                oversized_content,
                "application/pdf",
            )
        },
    )

    assert response.status_code == 413
    assert response.json()["detail"] == "PDF exceeds the 10 MB size limit."


def test_upload_rejects_corrupted_pdf():
    response = client.post(
        "/upload",
        files={
            "file": (
                "corrupted.pdf",
                b"%PDF-this is not a valid PDF document",
                "application/pdf",
            )
        },
    )

    assert response.status_code == 400
    assert "Unable to process PDF" in response.json()["detail"]
