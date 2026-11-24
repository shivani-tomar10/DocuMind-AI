
import re
from backend.vector_store import collection

results = collection.get(
    where={"source": "sample.pdf"},
    include=["documents", "metadatas"]
)

matches = []

for chunk_id, document, metadata in zip(
    results["ids"],
    results["documents"],
    results["metadatas"]
):
    # Remove repeated book title and page header
    cleaned_text = re.sub(
        r"\d+\s*•\s*Artificial Intelligence,\s*Machine Learning,\s*Deep Learning",
        "",
        document,
        flags=re.IGNORECASE
    )

    cleaned_text = cleaned_text.lower()

    if (
        "machine learning" in cleaned_text
        and "artificial intelligence" in cleaned_text
    ):
        matches.append({
            "chunk_id": chunk_id,
            "page": metadata["page_number"],
            "content": document
        })

print("\nMeaningful AI + ML matches:", len(matches))

for item in matches:
    print("\nChunk:", item["chunk_id"])
    print("Page:", item["page"])
    print("Content:\n", item["content"])
    print("-" * 70)
