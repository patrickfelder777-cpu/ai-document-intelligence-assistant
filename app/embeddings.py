
from functools import lru_cache

from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def get_embedding_model():
    """Load and cache the embedding model."""
    return SentenceTransformer(MODEL_NAME)


def generate_embeddings(texts: list[str]) -> list[list[float]]:
    """Convert a list of text strings into embeddings."""

    if not texts:
        return []

    model = get_embedding_model()

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    return embeddings.tolist()


def generate_query_embedding(query: str) -> list[float]:
    """Generate an embedding for a user's question."""

    if not query.strip():
        raise ValueError("Query cannot be empty.")

    return generate_embeddings([query])[0]
