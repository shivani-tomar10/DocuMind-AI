
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from pathlib import Path
from uuid import uuid4

from backend.retrieval import (
    retrieve_with_rrf,
    expand_retrieved_chunks
)

from backend.prompts import build_context

from backend.llm import (
    generate_search_query,
    generate_answer,
    validate_citations
)

from backend.pdf_processor import extract_text_from_pdf
from backend.chunking import create_chunks_with_metadata
from backend.embeddings import generate_embeddings
from backend.vector_store import store_chunks


# --------------------------------
# Application Configuration
# --------------------------------

app = FastAPI(
    title="DocuMind AI",
    description="Intelligent Multi-PDF Research Assistant",
    version="1.2.0"
)


# --------------------------------
# Upload Configuration
# --------------------------------

UPLOAD_FOLDER = Path("data/uploads")
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB


# --------------------------------
# Request Model
# --------------------------------

class QuestionRequest(BaseModel):
    question: str
    search_all_documents: bool = False


# --------------------------------
# GET: Home Endpoint
# --------------------------------

@app.get("/")
def home():
    return {
        "message": "Welcome to DocuMind AI!",
        "status": "API is running successfully"
    }


# --------------------------------
# POST: Ask Question Endpoint
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

        # Step 2: Generate alternative search query
        expanded_query = generate_search_query(query)

        # Step 3: Prepare multiple search queries
        search_queries = [query]

        if (
            expanded_query
            and expanded_query.strip()
            and expanded_query.strip().lower() != query.lower()
        ):
            search_queries.append(expanded_query.strip())

        # Step 4: Retrieve chunks using RRF
        results = retrieve_with_rrf(
            queries=search_queries,
            top_k=10,
            per_document=request.search_all_documents
        )

        # Step 5: Check retrieved documents
        documents = results.get("documents", [[]])

        if not documents or not documents[0]:
            raise HTTPException(
                status_code=404,
                detail=(
                    "No relevant documents found. "
                    "Please upload a PDF first."
                )
            )

        # Step 6: Expand retrieved chunks
        expanded_results = expand_retrieved_chunks(results)

        # Step 7: Construct context
        context = build_context(expanded_results)

        # Step 8: Generate answer using original question
        answer = generate_answer(query, context)

        # Step 9: Extract metadata from expanded results
        metadatas = expanded_results["metadatas"][0]

        # Step 10: Validate Gemini-generated citations
        answer, invalid_citations = validate_citations(
            answer,
            metadatas
        )

        # Step 11: Extract unique source and page metadata
        no_answer_message = (
            "The information was not found in the uploaded documents."
        )

        if answer.strip() == no_answer_message:
            # Hide irrelevant sources when the model cannot answer
            sources = []

        else:
            sources = []
            seen_sources = set()

            for metadata in metadatas:

                source_key = (
                    metadata["source"],
                    metadata["page_number"]
                )

                if source_key not in seen_sources:
                    sources.append(metadata)
                    seen_sources.add(source_key)

        # Step 12: Return response
        return {
            "question": query,
            "search_all_documents": request.search_all_documents,
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
# POST: PDF Upload Endpoint
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

        # Step 5: Create unique filename
        original_name = Path(file.filename).name

        unique_name = (
            f"{Path(original_name).stem}_{uuid4().hex[:8]}.pdf"
        )

        pdf_path = UPLOAD_FOLDER / unique_name

        # Step 6: Save uploaded file
        pdf_path.write_bytes(file_content)

        # Step 7: Extract PDF text
        pages = extract_text_from_pdf(pdf_path)

        if not pages or not any(
            page["text"].strip() for page in pages
        ):
            raise HTTPException(
                status_code=400,
                detail="Invalid PDF or no extractable text found."
            )

        # Step 8: Create chunks
        chunks = create_chunks_with_metadata(pages)

        if not chunks:
            raise HTTPException(
                status_code=400,
                detail="No text chunks could be created."
            )

        # Step 9: Generate embeddings
        embeddings = generate_embeddings(chunks)

        # Step 10: Store chunks in ChromaDB
        # Uses batch insertion from vector_store.py
        store_chunks(chunks, embeddings)

        # Step 11: Return upload details
        return {
            "message": "PDF uploaded and processed successfully!",
            "filename": unique_name,
            "pages_processed": len(pages),
            "chunks_created": len(chunks)
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
