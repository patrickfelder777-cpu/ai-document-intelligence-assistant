
from pathlib import Path

import pymupdf


def extract_text_from_pdf(file_path: str | Path) -> list[dict]:
    """
    Extract text from each page of a PDF.

    Returns:
        A list of dictionaries containing page numbers and text.
    """
    file_path = Path(file_path)

    if not file_path.is_file():
        raise FileNotFoundError(f"PDF not found: {file_path}")

    if file_path.suffix.lower() != ".pdf":
        raise ValueError("The provided file must be a PDF.")

    pages = []

    with pymupdf.open(file_path) as document:
        if document.needs_pass:
            raise ValueError("Password-protected PDFs are not supported.")

        for page_index, page in enumerate(document):
            text = page.get_text("text").strip()

            if text:
                pages.append(
                    {
                        "page_number": page_index + 1,
                        "text": text,
                    }
                )

    return pages
