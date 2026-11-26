
import requests
import streamlit as st


# ================================================
# BACKEND CONFIGURATION
# ================================================

API_URL = "http://127.0.0.1:8000"


# ================================================
# PAGE CONFIGURATION
# ================================================

st.set_page_config(
    page_title="DocuMind AI",
    page_icon="📚",
    layout="wide"
)


# ================================================
# SESSION STATE
# ================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "uploaded_documents" not in st.session_state:
    st.session_state.uploaded_documents = []

if "selected_document" not in st.session_state:
    st.session_state.selected_document = None


# ================================================
# CHAT RESET CALLBACK
# ================================================

def clear_chat_on_document_change():

    st.session_state.messages = []


# ================================================
# CUSTOM STYLING
# ================================================

st.markdown("""
<style>

.main-title {
    font-size: 38px;
    font-weight: 700;
}

.subtitle {
    color: #888888;
    font-size: 16px;
}

</style>
""", unsafe_allow_html=True)


# ================================================
# HEADER
# ================================================

st.markdown(
    '<div class="main-title">📚 DocuMind AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Your Intelligent Multi-PDF Research Assistant'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# ================================================
# SIDEBAR - DOCUMENT MANAGER
# ================================================

with st.sidebar:

    st.header("📄 Document Manager")

    uploaded_files = st.file_uploader(
        "Upload PDF documents",
        type=["pdf"],
        accept_multiple_files=True,
        help="You can upload multiple PDF documents."
    )

    if st.button(
        "Upload Documents",
        use_container_width=True
    ):

        if not uploaded_files:

            st.warning("Please select at least one PDF.")

        else:

            successful_uploads = 0

            progress = st.progress(0)

            for index, uploaded_file in enumerate(uploaded_files):

                try:

                    files = {
                        "file": (
                            uploaded_file.name,
                            uploaded_file.getvalue(),
                            "application/pdf"
                        )
                    }

                    with st.spinner(
                        f"Processing {uploaded_file.name}..."
                    ):

                        response = requests.post(
                            f"{API_URL}/upload",
                            files=files,
                            timeout=300
                        )

                    if response.ok:

                        result = response.json()

                        filename = result["filename"]

                        st.success(
                            f"Uploaded: {filename}"
                        )

                        if (
                            filename
                            not in st.session_state.uploaded_documents
                        ):

                            st.session_state.uploaded_documents.append(
                                filename
                            )

                        successful_uploads += 1

                    else:

                        try:

                            error_message = response.json().get(
                                "detail",
                                response.text
                            )

                        except ValueError:

                            error_message = response.text

                        st.error(
                            f"{uploaded_file.name}: {error_message}"
                        )

                except requests.RequestException as error:

                    st.error(
                        f"Upload failed for {uploaded_file.name}: {error}"
                    )

                progress.progress(
                    (index + 1) / len(uploaded_files)
                )

            if successful_uploads:

                st.success(
                    f"{successful_uploads} document(s) "
                    "processed successfully!"
                )

                st.rerun()

    st.divider()


    # ============================================
    # SELECT PDF DOCUMENT
    # ============================================

    st.subheader("📚 Select a PDF")

    if st.session_state.uploaded_documents:

        document_options = sorted(
            st.session_state.uploaded_documents
        )

        # Keep the selected document valid.
        if (
            st.session_state.selected_document
            not in document_options
        ):

            st.session_state.selected_document = (
                document_options[0]
            )

        selected_document = st.selectbox(
            "Choose a document before asking questions",
            options=document_options,
            key="selected_document",
            on_change=clear_chat_on_document_change
        )

        st.success(
            f"Selected: {selected_document}"
        )

    else:

        st.session_state.selected_document = None

        st.info(
            "Upload at least one PDF to start chatting."
        )

    st.divider()


    # ============================================
    # UPLOADED DOCUMENTS
    # ============================================

    st.subheader("📄 Uploaded Documents")

    if st.session_state.uploaded_documents:

        for document in sorted(
            st.session_state.uploaded_documents
        ):

            if document == st.session_state.selected_document:

                st.markdown(f"✅ **{document}**")

            else:

                st.write(f"📄 {document}")

    else:

        st.caption("No documents uploaded in this session yet.")

    st.divider()


    # ============================================
    # CLEAR CHAT
    # ============================================

    if st.button(
        "🗑️ Clear Chat",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.rerun()

    st.caption(
        "Powered by FastAPI • ChromaDB • Gemini"
    )


# ================================================
# BACKEND HEALTH CHECK
# ================================================

try:

    health_response = requests.get(
        f"{API_URL}/",
        timeout=5
    )

    if health_response.ok:

        st.success(
            "Backend connected successfully."
        )

    else:

        st.warning(
            "Backend is responding with an error."
        )

except requests.RequestException:

    st.error(
        "Backend is not connected. "
        "Please check that FastAPI is running."
    )


# ================================================
# MAIN CHAT INTERFACE
# ================================================

st.subheader("💬 Chat with Your Selected Document")

if st.session_state.selected_document:

    st.caption(
        f"Currently chatting with: "
        f"{st.session_state.selected_document}"
    )

else:

    st.warning(
        "Please upload and select a PDF before asking questions."
    )


# ================================================
# DISPLAY CHAT HISTORY
# ================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        if (
            message["role"] == "assistant"
            and message.get("sources")
        ):

            with st.expander("📌 View Sources"):

                for source in message["sources"]:

                    st.write(
                        f"📄 {source['source']} "
                        f"| Page {source['page_number']}"
                    )


# ================================================
# CHAT INPUT
# ================================================

question = st.chat_input(
    "Ask something about your selected PDF...",
    disabled=not st.session_state.selected_document
)


# ================================================
# PROCESS QUESTION
# ================================================

if question:

    selected_document = st.session_state.selected_document

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):

        st.markdown(question)

    with st.chat_message("assistant"):

        with st.spinner(
            f"Searching {selected_document} and generating answer..."
        ):

            try:

                response = requests.post(
                    f"{API_URL}/ask",
                    json={
                        "question": question,
                        "selected_document": selected_document
                    },
                    timeout=300
                )

                if response.ok:

                    result = response.json()

                    answer = result["answer"]

                    sources = result.get("sources", [])

                    st.markdown(answer)

                    # DISPLAY SOURCES

                    if sources:

                        with st.expander("📌 View Sources"):

                            for source in sources:

                                st.write(
                                    f"📄 {source['source']} "
                                    f"| Page {source['page_number']}"
                                )

                    else:

                        st.info(
                            "No source metadata was returned."
                        )

                    # SAVE ASSISTANT RESPONSE

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })

                else:

                    try:

                        error_message = response.json().get(
                            "detail",
                            response.text
                        )

                    except ValueError:

                        error_message = response.text

                    st.error(
                        f"API Error {response.status_code}: "
                        f"{error_message}"
                    )

            except requests.RequestException as error:

                st.error(
                    f"Could not connect to the backend: {error}"
                )
