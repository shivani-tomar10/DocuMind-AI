
import hashlib
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel

from backend.retrieval import (
    retrieve_with_rrf,
    expand_retrieved_chunks
)

from backend.prompts import build_context

from backend.llm import (
    generate_search_queries,
    generate_answer,
    validate_citations
)

from backend.pdf_processor import extract_text_from_pdf
from backend.chunking import create_chunks_with_metadata
from backend.embeddings import generate_embeddings
from backend.vector_store import store_chunks, collection


# --------------------------------
# 1. Application Configuration
# --------------------------------

app = FastAPI(
    title="DocuMind AI",
    description="Intelligent Multi-PDF Research Assistant",
    version="1.4.0"
)


# --------------------------------
# 2. Upload Configuration
# --------------------------------

UPLOAD_FOLDER = Path("data/uploads")
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB


# --------------------------------
# 3. Request Model
# --------------------------------

class QuestionRequest(BaseModel):
    question: str
    selected_document: str


# --------------------------------
# 4. GET: Home Endpoint
# --------------------------------

@app.get("/")
def home():

    return {
        "message": "Welcome to DocuMind AI!",
        "status": "API is running successfully"
    }


# --------------------------------
# 5. POST: Ask Question Endpoint
# --------------------------------

@app.post("/ask")
def ask_question(request: QuestionRequest):

    try:

        # Step 1: Validate question
        query = request.question.strip()

        if not query:
            raise HTTPException(
                status_code=400,
                detail="Question cannot be empty."
            )

        # Step 2: Validate selected document
        selected_document = Path(
            request.selected_document
        ).name

        if not selected_document.lower().endswith(".pdf"):
            raise HTTPException(
                status_code=400,
                detail="Please select a valid PDF document."
            )

        # Step 3: Check whether selected PDF exists
        if not (UPLOAD_FOLDER / selected_document).is_file():

            raise HTTPException(
                status_code=404,
                detail="Selected PDF was not found."
            )

        # Step 4: Check whether selected PDF is indexed
        indexed_documents = collection.get(
            where={"source": selected_document},
            include=["metadatas"]
        )

        if not indexed_documents["ids"]:

            raise HTTPException(
                status_code=404,
                detail=(
                    "Selected PDF has not been indexed. "
                    "Please upload it again."
                )
            )

        # Step 5: Generate alternative search queries
        alternative_queries = generate_search_queries(
            query,
            num_queries=3
        )

        # Step 6: Combine original and alternative queries
        search_queries = [query]

        for alternative in alternative_queries:

            cleaned_query = alternative.strip()

            if (
                cleaned_query
                and cleaned_query.lower()
                not in {
                    item.lower()
                    for item in search_queries
                }
            ):
                search_queries.append(cleaned_query)

        # Step 7: Retrieve chunks ONLY from selected PDF
        results = retrieve_with_rrf(
            queries=search_queries,
            top_k=10,
            candidate_k=40,
            per_document=False,
            source=selected_document
        )

        # Step 8: Check retrieved documents
        documents = results.get("documents", [[]])

        if not documents or not documents[0]:

            raise HTTPException(
                status_code=404,
                detail=(
                    "No relevant information found in "
                    "the selected PDF. Try another question."
                )
            )

        # Step 9: Expand retrieved chunks
        expanded_results = expand_retrieved_chunks(
            results
        )

        # Step 10: Construct document context
        context = build_context(
            expanded_results
        )

        # Step 11: Generate answer
        answer = generate_answer(
            query,
            context
        )

        # Step 12: Extract metadata
        metadatas = expanded_results["metadatas"][0]

        # Step 13: Validate citations
        answer, invalid_citations = validate_citations(
            answer,
            metadatas
        )

        # Step 14: Extract unique sources and pages
        no_answer_message = (
            "The information was not found in the uploaded documents."
        )

        if answer.strip() == no_answer_message:

            sources = []

        else:

            sources = []
            seen_sources = set()

            for metadata in metadatas:

                source_key = (
                    metadata["source"],
                    metadata["page_number"]
                )

                if (
                    metadata["source"] == selected_document
                    and source_key not in seen_sources
                ):

                    sources.append(metadata)
                    seen_sources.add(source_key)

        # Step 15: Return response
        return {
            "question": query,
            "selected_document": selected_document,
            "answer": answer,
            "sources": sources,
            "invalid_citations": invalid_citations,
            "search_queries": search_queries,
            "retrieved_chunks": len(results["ids"][0]),
            "expanded_chunks": len(expanded_results["ids"][0])
        }

    except HTTPException:
        raise

    except Exception as error:

        print(
            "Question processing failed:",
            type(error).__name__,
            str(error)
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Something went wrong while "
                "processing your question."
            )
        )


# --------------------------------
# 6. POST: PDF Upload Endpoint
# --------------------------------

@app.post("/upload")
def upload_pdf(file: UploadFile = File(...)):

    pdf_path = None

    try:

        # Step 1: Validate filename
        if (
            not file.filename
            or not file.filename.lower().endswith(".pdf")
        ):

            raise HTTPException(
                status_code=400,
                detail="Only PDF files are allowed."
            )

        # Step 2: Read uploaded file
        file_content = file.file.read()

        # Step 3: Check empty file
        if not file_content:

            raise HTTPException(
                status_code=400,
                detail="Uploaded PDF is empty."
            )

        # Step 4: Check file size
        if len(file_content) > MAX_FILE_SIZE:

            raise HTTPException(
                status_code=413,
                detail="File size exceeds the 10 MB limit."
            )

        # Step 5: Calculate SHA-256 hash
        file_hash = hashlib.sha256(
            file_content
        ).hexdigest()

        # Step 6: Check for an existing identical PDF
        for existing_file in UPLOAD_FOLDER.glob("*.pdf"):

            existing_hash = hashlib.sha256(
                existing_file.read_bytes()
            ).hexdigest()

            if existing_hash == file_hash:

                return {
                    "message": "This PDF has already been uploaded.",
                    "filename": existing_file.name,
                    "duplicate": True,
                    "sha256": file_hash
                }

        # Step 7: Create unique filename
        original_name = Path(file.filename).name

        unique_name = (
            f"{Path(original_name).stem}_{uuid4().hex[:8]}.pdf"
        )

        pdf_path = UPLOAD_FOLDER / unique_name

        # Step 8: Save uploaded file
        pdf_path.write_bytes(file_content)

        # Step 9: Extract PDF text
        pages = extract_text_from_pdf(pdf_path)

        if not pages or not any(
            page["text"].strip()
            for page in pages
        ):

            raise HTTPException(
                status_code=400,
                detail="Invalid PDF or no extractable text found."
            )

        # Step 10: Create chunks
        chunks = create_chunks_with_metadata(pages)

        if not chunks:

            raise HTTPException(
                status_code=400,
                detail="No text chunks could be created."
            )

        # Step 11: Generate embeddings
        embeddings = generate_embeddings(chunks)

        # Step 12: Store chunks in ChromaDB
        store_chunks(
            chunks,
            embeddings
        )

        # Step 13: Return upload details
        return {
            "message": "PDF uploaded and processed successfully!",
            "filename": unique_name,
            "pages_processed": len(pages),
            "chunks_created": len(chunks),
            "duplicate": False,
            "sha256": file_hash
        }

    except HTTPException:

        if pdf_path and pdf_path.exists():
            pdf_path.unlink()

        raise

    except Exception as error:

        print(
            "PDF processing failed:",
            type(error).__name__,
            str(error)
        )

        if pdf_path and pdf_path.exists():
            pdf_path.unlink()

        raise HTTPException(
            status_code=500,
            detail=(
                "PDF processing failed. "
                "Please check the file and try again."
            )
        )

    finally:
        file.file.close()
