
from backend.pdf_processor import process_multiple_pdfs
from backend.chunking import create_chunks_with_metadata
from backend.embeddings import generate_embeddings
from backend.vector_store import store_chunks
from backend.retrieval import (
    retrieve_relevant_chunks,
    expand_retrieved_chunks
)
from backend.prompts import build_context
from backend.llm import generate_answer

if __name__ == "__main__":

    # Step 1: Extract PDF pages
    pages = process_multiple_pdfs("data/uploads")

    print("\nTotal extracted pages:", len(pages))

    # Step 2: Create chunks with metadata
    chunks = create_chunks_with_metadata(pages)

    print("Total chunks created:", len(chunks))

    # Step 3: Generate embeddings
    embeddings = generate_embeddings(chunks)

    print("Total embeddings:", len(embeddings))
    print("Embedding dimensions:", embeddings.shape)

    # Step 4: Store chunks in ChromaDB
    store_chunks(chunks, embeddings)

    # Step 5: Ask the user a question
    query = input("\nAsk a question about your PDF: ")

    # Step 6: Retrieve relevant chunks
    results = retrieve_relevant_chunks(
        query,
        top_k=3
    )

    print("\nOriginal Retrieved Chunks:")
    print("Total:", len(results["documents"][0]))

    # Step 7: Expand retrieved chunks
    expanded_results = expand_retrieved_chunks(results)

    print("\nExpanded Retrieved Chunks:")
    print("Total:", len(expanded_results["documents"][0]))

    # Step 8: Construct context using expanded chunks
    context = build_context(expanded_results)

    print("\nRetrieved Context:")
    print(context)

    # Step 9: Generate answer using Gemini
    answer = generate_answer(query, context)

    print("\nUser Question:", query)
    print("\nDocuMind AI Answer:", answer)
