
import re


CITATION_PATTERN = re.compile(
    r"\[Source\s+(\d+),\s*Page\s+(\d+)\]",
    re.IGNORECASE,
)


def validate_citations(
    answer: str,
    sources: list[dict],
) -> dict:
    """
    Check whether citations reference retrieved sources
    and match their actual page numbers.
    """

    citations = CITATION_PATTERN.findall(answer)

    valid_citations = []
    invalid_citations = []

    for source_number, page_number in citations:
        source_index = int(source_number) - 1
        cited_page = int(page_number)

        if 0 <= source_index < len(sources):
            source = sources[source_index]

            if source.get("page_number") == cited_page:
                valid_citations.append({
                    "source_number": int(source_number),
                    "page_number": cited_page,
                    "document_id": source.get("document_id"),
                })
                continue

        invalid_citations.append({
            "source_number": int(source_number),
            "page_number": cited_page,
        })

    return {
        "total_citations": len(citations),
        "valid_citations": valid_citations,
        "invalid_citations": invalid_citations,
        "citations_valid": (
            len(invalid_citations) == 0
            and len(valid_citations) > 0
        ),
    }
