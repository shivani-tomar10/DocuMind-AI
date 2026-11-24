
from backend.vector_store import collection

search_terms = [
    "ACID",
    "Atomicity",
    "Consistency",
    "Isolation",
    "Durability"
]

print("\n========== ACID COVERAGE CHECK ==========")

for term in search_terms:
    print(f"\nSearching for: {term}")

    results = collection.get(
        where={"source": "31. SQL Notes by Apna College_61abe913.pdf"},
        include=["documents", "metadatas"]
    )

    matches = []

    for document, metadata in zip(
        results["documents"], results["metadatas"]
    ):
        if term.lower() in document.lower():
            matches.append((document, metadata))

    print("Matching chunks:", len(matches))

    for document, metadata in matches[:3]:
        print("\nPage:", metadata["page_number"])
        print("Content:", document[:500])
        print("-" * 40)

print("\n========== CHECK COMPLETED ==========")
