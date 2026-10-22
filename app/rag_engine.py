from app.llm_client import generate_answer
from app.vector_store import VectorStore
from app.citation_validator import validate_citations


def build_context(results: list[dict]) -> str:
    """Format retrieved PDF passages with source citations."""

    if not results:
        return ""

    context_parts = []

    for index, result in enumerate(results, start=1):
        document_id = result["document_id"]
        page_number = result["page_number"]
        text = result["text"]

        context_parts.append(
            f"[Source {index}]\n"
            f"Document ID: {document_id}\n"
            f"Page: {page_number}\n"
            f"Content: {text}"
        )

    return "\n\n".join(context_parts)


def retrieve_context(
    question: str,
    document_id: str | None = None,
    top_k: int = 3,
) -> dict:
    """Retrieve relevant PDF passages and prepare LLM context."""

    if not question.strip():
        raise ValueError("Question cannot be empty.")

    vector_store = VectorStore()

    results = vector_store.search(
        query=question,
        top_k=top_k,
        document_id=document_id,
    )

    return {
        "question": question,
        "context": build_context(results),
        "sources": results,
        "total_sources": len(results),
    }


def answer_question(
    question: str,
    document_id: str | None = None,
    top_k: int = 3,
) -> dict:
    """
    Retrieve PDF passages, generate an AI answer,
    and validate the citations.
    """

    if not question.strip():
        raise ValueError("Question cannot be empty.")

    retrieved = retrieve_context(
        question=question,
        document_id=document_id,
        top_k=top_k,
    )

    answer = generate_answer(
        question=question,
        context=retrieved["context"],
    )

    citation_report = validate_citations(
        answer=answer,
        sources=retrieved["sources"],
    )

    return {
        "question": question,
        "answer": answer,
        "document_id": document_id,
        "total_sources": retrieved["total_sources"],
        "sources": retrieved["sources"],
        "citation_validation": citation_report,
    }
