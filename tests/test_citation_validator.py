
import pytest

from app.citation_validator import validate_citations


@pytest.fixture
def sample_sources():
    return [
        {
            "document_id": "document_A",
            "page_number": 2,
            "text": "Machine learning identifies patterns.",
        },
        {
            "document_id": "document_B",
            "page_number": 5,
            "text": "Docker packages applications.",
        },
    ]


def test_valid_citation(sample_sources):
    answer = "Machine learning identifies patterns [Source 1, Page 2]."

    result = validate_citations(answer, sample_sources)

    assert result["citations_valid"] is True
    assert result["total_citations"] == 1
    assert len(result["valid_citations"]) == 1
    assert result["invalid_citations"] == []


def test_multiple_valid_citations(sample_sources):
    answer = (
        "Machine learning identifies patterns [Source 1, Page 2]. "
        "Docker packages applications [Source 2, Page 5]."
    )

    result = validate_citations(answer, sample_sources)

    assert result["citations_valid"] is True
    assert result["total_citations"] == 2
    assert len(result["valid_citations"]) == 2


def test_nonexistent_source(sample_sources):
    answer = "This information is documented [Source 3, Page 1]."

    result = validate_citations(answer, sample_sources)

    assert result["citations_valid"] is False
    assert len(result["invalid_citations"]) == 1


def test_incorrect_page_number(sample_sources):
    answer = "Machine learning identifies patterns [Source 1, Page 9]."

    result = validate_citations(answer, sample_sources)

    assert result["citations_valid"] is False
    assert result["invalid_citations"][0]["page_number"] == 9


def test_missing_citations(sample_sources):
    answer = "Machine learning identifies patterns."

    result = validate_citations(answer, sample_sources)

    assert result["total_citations"] == 0
    assert result["citations_valid"] is False


def test_mixed_valid_and_invalid_citations(sample_sources):
    answer = (
        "Machine learning identifies patterns [Source 1, Page 2]. "
        "Docker information [Source 2, Page 99]."
    )

    result = validate_citations(answer, sample_sources)

    assert result["citations_valid"] is False
    assert len(result["valid_citations"]) == 1
    assert len(result["invalid_citations"]) == 1


def test_citation_preserves_document_id(sample_sources):
    answer = "Docker packages applications [Source 2, Page 5]."

    result = validate_citations(answer, sample_sources)

    assert result["valid_citations"][0]["document_id"] == "document_B"


def test_case_insensitive_citations(sample_sources):
    answer = "Machine learning identifies patterns [source 1, page 2]."

    result = validate_citations(answer, sample_sources)

    assert result["citations_valid"] is True
