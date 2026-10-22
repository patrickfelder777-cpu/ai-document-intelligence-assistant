
from pathlib import Path
from uuid import uuid4

import chromadb

from app.embeddings import (
    generate_embeddings,
    generate_query_embedding,
)


class VectorStore:
    """Store and retrieve document chunks using ChromaDB."""

    def __init__(self, persist_directory="data/chroma_db"):
        Path(persist_directory).mkdir(
            parents=True,
            exist_ok=True,
        )

        self.client = chromadb.PersistentClient(
            path=str(persist_directory)
        )

        self.collection = self.client.get_or_create_collection(
            name="document_chunks",
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(self, chunks: list[dict], document_id: str):
        """Generate embeddings and store document chunks."""

        if not chunks:
            return 0

        texts = [chunk["text"] for chunk in chunks]
        embeddings = generate_embeddings(texts)

        ids = [
            f"{document_id}_{chunk['chunk_id']}_{uuid4().hex[:8]}"
            for chunk in chunks
        ]

        metadatas = [
            {
                "document_id": document_id,
                "page_number": chunk["page_number"],
                "chunk_id": chunk["chunk_id"],
            }
            for chunk in chunks
        ]

        self.collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        return len(chunks)

    def search(
        self,
        query: str,
        top_k: int = 3,
        document_id: str | None = None,
    ):
        """Search document chunks, optionally filtering by document ID."""

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        if not query.strip():
            raise ValueError("Query cannot be empty")

        where_filter = (
            {"document_id": document_id}
            if document_id
            else None
        )

        if where_filter:
            matching_count = self.collection.count()
        else:
            matching_count = self.collection.count()

        if matching_count == 0:
            return []

        query_embedding = generate_query_embedding(query)

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, matching_count),
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        matches = []

        for text, metadata, distance in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            matches.append(
                {
                    "text": text,
                    "document_id": metadata["document_id"],
                    "page_number": metadata["page_number"],
                    "chunk_id": metadata["chunk_id"],
                    "distance": distance,
                }
            )

        return matches

