# AI Document Intelligence Assistant

## Project Demo

Watch my AI Document Intelligence Assistant in action:

[▶ Watch the AI Document Intelligence Demo](./AI_Document_Intelligence_Demo.mp4)

This demonstration showcases PDF uploading, AI-powered document questions, semantic search, and source citations.

An AI-powered document question-answering application built with Python, FastAPI, Streamlit, ChromaDB, and Retrieval-Augmented Generation (RAG).

Upload a PDF, ask questions about its contents, and receive AI-generated answers with page-level source citations.

## Features

- **PDF Processing:** Extracts text from uploaded PDF documents using PyMuPDF.
- **Document Chunking:** Splits extracted text into overlapping chunks for retrieval.
- **Semantic Search:** Uses sentence-transformer embeddings and ChromaDB to find relevant passages.
- **Retrieval-Augmented Generation:** Supplies retrieved document context to a large language model through the Nebius API.
- **Source Citations:** Returns document references and page numbers alongside AI-generated answers.
- **Citation Validation:** Checks whether generated citation references correspond to retrieved sources.
- **Interactive Interface:** Streamlit application for uploading PDFs and asking questions.
- **REST API:** FastAPI endpoints for document upload, search, and question answering.
- **Containerization:** Docker Compose runs the API and Streamlit frontend together.
- **Automated Testing:** Pytest suite with 72 passing tests and GitHub Actions CI.

## Technology Stack

| Category | Technologies |
|---|---|
| Programming language | Python 3.11 |
| Backend | FastAPI, Uvicorn |
| Frontend | Streamlit |
| PDF extraction | PyMuPDF |
| Embeddings | Sentence Transformers, all-MiniLM-L6-v2 |
| Vector database | ChromaDB |
| LLM provider | Nebius Token Factory |
| Testing | Pytest |
| Deployment | Docker, Docker Compose |
| CI/CD | GitHub Actions |

## System Architecture

The application uses a retrieval-augmented generation pipeline.

**Document ingestion:**

1. User uploads a PDF through Streamlit.
2. FastAPI extracts the document text.
3. Text is divided into overlapping chunks.
4. Sentence Transformers generates vector embeddings.
5. ChromaDB stores the chunks, embeddings, and document metadata.

**Question answering:**

1. User submits a question.
2. The application generates an embedding for the question.
3. ChromaDB retrieves relevant document passages.
4. The retrieved context is passed to the Nebius-hosted LLM.
5. The application returns an answer with source references and citation-validation results.

Streamlit communicates with FastAPI through HTTP requests.

## Getting Started

### Prerequisites

- Docker Desktop
- Docker Compose
- A Nebius API key

### 1. Clone the Repository

```bash
git clone https://github.com/patrickfelder777-cpu/ai-document-intelligence-assistant.git
cd ai-document-intelligence-assistant
```

### 2. Configure Environment Variables

Create a `.env` file in the project root:

```env
NEBIUS_API_KEY=your

