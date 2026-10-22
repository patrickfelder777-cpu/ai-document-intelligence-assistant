
import pytest
import pymupdf

from app.pdf_processor import extract_text_from_pdf


def test_extract_text_from_pdf(tmp_path):
    pdf_path = tmp_path / "sample.pdf"

    document = pymupdf.open()

    page1 = document.new_page()
    page1.insert_text((72, 72), "Machine learning uses data.")

    page2 = document.new_page()
    page2.insert_text((72, 72), "RAG retrieves relevant information.")

    document.save(pdf_path)
    document.close()

    pages = extract_text_from_pdf(pdf_path)

    assert len(pages) == 2
    assert pages[0]["page_number"] == 1
    assert "Machine learning uses data." in pages[0]["text"]
    assert pages[1]["page_number"] == 2
    assert "RAG retrieves relevant information." in pages[1]["text"]


def test_missing_pdf_raises_error(tmp_path):
    missing_file = tmp_path / "missing.pdf"

    with pytest.raises(FileNotFoundError):
        extract_text_from_pdf(missing_file)


def test_non_pdf_file_raises_error(tmp_path):
    text_file = tmp_path / "sample.txt"
    text_file.write_text("This is not a PDF.")

    with pytest.raises(ValueError, match="must be a PDF"):
        extract_text_from_pdf(text_file)


def test_empty_pdf_returns_no_pages(tmp_path):
    pdf_path = tmp_path / "empty.pdf"

    document = pymupdf.open()
    document.new_page()
    document.save(pdf_path)
    document.close()

    pages = extract_text_from_pdf(pdf_path)

    assert pages == []
