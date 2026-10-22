
def chunk_document(
    pages: list[dict],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[dict]:
    """
    Split extracted PDF pages into overlapping text chunks.

    Each chunk retains its original page number.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")

    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError(
            "chunk_overlap must be non-negative and smaller than chunk_size"
        )

    chunks = []
    chunk_id = 0
    step = chunk_size - chunk_overlap

    for page in pages:
        text = page["text"]
        page_number = page["page_number"]

        if not text.strip():
            continue

        for start in range(0, len(text), step):
            chunk_text = text[start:start + chunk_size]

            if not chunk_text.strip():
                continue

            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "page_number": page_number,
                    "text": chunk_text,
                }
            )

            chunk_id += 1

            if start + chunk_size >= len(text):
                break

    return chunks
