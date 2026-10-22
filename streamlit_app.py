
import os

import requests
import streamlit as st


API_URL = os.getenv(
    "API_URL",
    "http://127.0.0.1:8000",
)

MAX_FILE_SIZE = 10 * 1024 * 1024


st.set_page_config(
    page_title="AI Document Intelligence",
    page_icon="📄",
    layout="wide",
)


if "document_id" not in st.session_state:
    st.session_state.document_id = None

if "document_name" not in st.session_state:
    st.session_state.document_name = None

if "messages" not in st.session_state:
    st.session_state.messages = []


st.title("AI Document Intelligence Assistant")

st.caption(
    "Upload PDF documents, ask questions, "
    "and receive AI-generated answers with source citations."
)


with st.sidebar:
    st.header("Document Management")

    uploaded_file = st.file_uploader(
        "Upload a PDF",
        type=["pdf"],
        help="Maximum file size: 10 MB",
    )

    if uploaded_file is not None:
        if uploaded_file.size > MAX_FILE_SIZE:
            st.error("PDF exceeds the 10 MB upload limit.")

        elif st.button(
            "Process Document",
            use_container_width=True,
        ):
            with st.spinner("Processing and indexing PDF..."):
                try:
                    response = requests.post(
                        f"{API_URL}/upload",
                        files={
                            "file": (
                                uploaded_file.name,
                                uploaded_file.getvalue(),
                                "application/pdf",
                            )
                        },
                        timeout=120,
                    )

                    response.raise_for_status()
                    result = response.json()

                    st.session_state.document_id = (
                        result["document_id"]
                    )

                    st.session_state.document_name = (
                        uploaded_file.name
                    )

                    st.session_state.messages = []

                    st.success("Document processed successfully!")

                except requests.exceptions.RequestException:
                    st.error(
                        "Unable to upload the document. "
                        "Check that the FastAPI server is running."
                    )

    st.divider()

    st.subheader("Current Document")

    if st.session_state.document_id:
        st.success(st.session_state.document_name)

        st.caption(
            f"Document ID: {st.session_state.document_id}"
        )

        if st.button(
            "Clear Document",
            use_container_width=True,
        ):
            st.session_state.document_id = None
            st.session_state.document_name = None
            st.session_state.messages = []
            st.rerun()
    else:
        st.info("No document selected.")

    st.divider()

    st.subheader("Search Settings")

    top_k = st.slider(
        "Number of sources",
        min_value=1,
        max_value=10,
        value=3,
    )


st.subheader("Ask Questions About Your PDF")

if not st.session_state.document_id:
    st.info(
        "Upload and process a PDF to begin asking questions."
    )

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message.get("sources"):
            with st.expander("View Supporting Sources"):
                for index, source in enumerate(
                    message["sources"],
                    start=1,
                ):
                    st.markdown(
                        f"**Source {index} — "
                        f"Page {source['page_number']}**"
                    )

                    st.write(source["text"])

                    st.divider()

        if message.get("citation_validation"):
            report = message["citation_validation"]

            if report["citations_valid"]:
                st.caption(
                    "Citation references match retrieved sources."
                )
            else:
                st.warning(
                    "Some citation references are missing "
                    "or could not be validated."
                )


question = st.chat_input(
    "Ask a question about your PDF...",
    disabled=not bool(st.session_state.document_id),
)

if question:
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing document..."):
            try:
                response = requests.post(
                    f"{API_URL}/ask",
                    json={
                        "question": question,
                        "document_id": (
                            st.session_state.document_id
                        ),
                        "top_k": top_k,
                    },
                    timeout=120,
                )

                response.raise_for_status()
                result = response.json()

                answer = result["answer"]
                sources = result.get("sources", [])
                citation_validation = result.get(
                    "citation_validation",
                    {},
                )

                st.markdown(answer)

                if sources:
                    with st.expander("View Supporting Sources"):
                        for index, source in enumerate(
                            sources,
                            start=1,
                        ):
                            st.markdown(
                                f"**Source {index} — "
                                f"Page {source['page_number']}**"
                            )

                            st.write(source["text"])
                            st.divider()

                if citation_validation:
                    if citation_validation.get(
                        "citations_valid"
                    ):
                        st.caption(
                            "Citation references match "
                            "retrieved sources."
                        )
                    else:
                        st.warning(
                            "Some citation references "
                            "could not be validated."
                        )

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                        "citation_validation": citation_validation,
                    }
                )

            except requests.exceptions.RequestException:
                st.error(
                    "Unable to generate an answer. "
                    "Check the FastAPI server and try again."
                )
