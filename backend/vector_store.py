
import os
from pathlib import Path

import chromadb


# --------------------------------
# ChromaDB Configuration
# --------------------------------

CHROMA_DB_PATH = os.getenv(
    "CHROMA_DB_PATH",
    "data/chroma_db"
)

Path(CHROMA_DB_PATH).mkdir(
    parents=True,
    exist_ok=True
)

client = chromadb.PersistentClient(
    path=CHROMA_DB_PATH
)

collection = client.get_or_create_collection(
    name="documind_documents",
    embedding_function=None
)


# --------------------------------
# Store Chunks in Batches
# --------------------------------

def store_chunks(chunks, embeddings):

    ids = [
        chunk["chunk_id"]
        for chunk in chunks
    ]

    documents = [
        chunk["text"]
        for chunk in chunks
    ]

    metadatas = [
        {
            "source": chunk["source"],
            "page_number": chunk["page_number"]
        }
        for chunk in chunks
    ]

    max_batch_size = client.get_max_batch_size()

    batch_size = min(max_batch_size, 1000)

    total_chunks = len(chunks)

    print(f"Total chunks to store: {total_chunks}")
    print(f"Batch size: {batch_size}")

    for start in range(0, total_chunks, batch_size):

        end = min(start + batch_size, total_chunks)

        collection.upsert(
            ids=ids[start:end],
            documents=documents[start:end],
            embeddings=embeddings[start:end].tolist(),
            metadatas=metadatas[start:end]
        )

        print(
            f"Stored batch: {start + 1} to {end} "
            f"of {total_chunks}"
        )

    print(
        "Total documents stored:",
        collection.count()
    )


# --------------------------------
# List Indexed PDF Documents
# --------------------------------

def list_indexed_documents():

    documents = {}

    total_chunks = collection.count()

    batch_size = 1000

    for offset in range(0, total_chunks, batch_size):

        results = collection.get(
            limit=batch_size,
            offset=offset,
            include=["metadatas"]
        )

        metadatas = results.get("metadatas", [])

        for metadata in metadatas:

            if not metadata:
                continue

            filename = metadata.get("source")

            if filename:

                documents[filename] = (
                    documents.get(filename, 0) + 1
                )

    return [
        {
            "filename": filename,
            "chunks": chunks
        }
        for filename, chunks in sorted(documents.items())
    ]
