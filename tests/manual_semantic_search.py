
from uuid import uuid4

from app.vector_store import VectorStore


store = VectorStore()

document_id = f"semantic_test_{uuid4().hex[:8]}"

chunks = [
    {
        "chunk_id": 0,
        "page_number": 1,
        "text": (
            "Machine learning algorithms identify patterns "
            "in data and make predictions."
        ),
    },
    {
        "chunk_id": 1,
        "page_number": 2,
        "text": (
            "Docker containers package applications "
            "with their dependencies for deployment."
        ),
    },
    {
        "chunk_id": 2,
        "page_number": 3,
        "text": (
            "Vector databases store embeddings and "
            "support semantic similarity search."
        ),
    },
]

added = store.add_chunks(chunks, document_id=document_id)

print("Chunks added:", added)
print("Total stored chunks:", store.collection.count())

question = "How can I find similar information using embeddings?"

results = store.search(question, top_k=3)

print("\nQuestion:", question)

for index, result in enumerate(results, start=1):
    print(f"\nResult {index}")
    print("Page:", result["page_number"])
    print("Text:", result["text"])
    print("Distance:", round(result["distance"], 4))
