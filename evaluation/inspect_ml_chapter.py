

from backend.retrieval import retrieve_relevant_chunks

queries = [
    "Machine Learning is a subset of Artificial Intelligence",
    "How is Machine Learning related to AI?",
    "Machine Learning uses data to learn and perform tasks as a part of AI"
]

for query in queries:

    print("\nQUERY:", query)
    print("=" * 70)

    results = retrieve_relevant_chunks(query, top_k=5)

    for rank, (chunk_id, document, metadata, distance) in enumerate(
        zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        ),
        start=1
    ):
        print(f"\nRank: {rank}")
        print("Chunk:", chunk_id)
        print("Page:", metadata["page_number"])
        print("Distance:", round(distance, 4))
        print("Content:", document[:500])
