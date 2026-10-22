
import pytest

from app.chunking import chunk_document


def test_chunk_short_document():
    pages = [
        {"page_number": 1, "text": "Machine learning is useful."}
    ]

    chunks = chunk_document(pages)

    assert len(chunks) == 1
    assert chunks[0]["chunk_id"] == 0
    assert chunks[0]["page_number"] == 1
    assert chunks[0]["text"] == "Machine learning is useful."


def test_chunk_overlap():
    pages = [
        {"page_number": 1, "text": "ABCDEFGHIJKLMNOPQRSTUVWXYZ"}
    ]

    chunks = chunk_document(
        pages,
        chunk_size=10,
        chunk_overlap=3,
    )

    assert len(chunks) == 4
    assert chunks[0]["text"] == "ABCDEFGHIJ"
    assert chunks[1]["text"] == "HIJKLMNOPQ"
    assert chunks[2]["text"] == "OPQRSTUVWX"
    assert chunks[3]["text"] == "VWXYZ"

    assert chunks[0]["text"][-3:] == chunks[1]["text"][:3]


def test_preserves_page_numbers():
    pages = [
        {"page_number": 1, "text": "First page."},
        {"page_number": 2, "text": "Second page."},
    ]

    chunks = chunk_document(pages)

    assert len(chunks) == 2
    assert chunks[0]["page_number"] == 1
    assert chunks[1]["page_number"] == 2


def test_unique_chunk_ids():
    pages = [
        {"page_number": 1, "text": "A" * 25},
        {"page_number": 2, "text": "B" * 25},
    ]

    chunks = chunk_document(
        pages,
        chunk_size=10,
        chunk_overlap=2,
    )

    chunk_ids = [chunk["chunk_id"] for chunk in chunks]

    assert len(chunk_ids) == len(set(chunk_ids))
    assert chunk_ids == list(range(len(chunks)))


def test_empty_document():
    assert chunk_document([]) == []


def test_skips_blank_pages():
    pages = [
        {"page_number": 1, "text": "   "},
        {"page_number": 2, "text": "Useful information."},
    ]

    chunks = chunk_document(pages)

    assert len(chunks) == 1
    assert chunks[0]["page_number"] == 2


def test_invalid_chunk_size():
    with pytest.raises(ValueError, match="chunk_size"):
        chunk_document(
            [{"page_number": 1, "text": "Example"}],
            chunk_size=0,
        )


def test_invalid_chunk_overlap():
    with pytest.raises(ValueError, match="chunk_overlap"):
        chunk_document(
            [{"page_number": 1, "text": "Example"}],
            chunk_size=10,
            chunk_overlap=10,
        )


def test_negative_chunk_overlap():
    with pytest.raises(ValueError, match="chunk_overlap"):
        chunk_document(
            [{"page_number": 1, "text": "Example"}],
            chunk_size=10,
            chunk_overlap=-1,
        )
