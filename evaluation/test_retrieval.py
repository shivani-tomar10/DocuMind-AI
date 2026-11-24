
from backend.retrieval import retrieve_with_rrf


# ==================================================
# 1. EVALUATION QUESTIONS
# ==================================================

test_questions = [
    "What is Artificial Intelligence?",
    "What are the goals of Artificial Intelligence?",
    "Explain Strong AI versus Weak AI.",
    "What is the Turing Test?",
    "What are the major parts of AI?"
]


# ==================================================
# 2. RETRIEVAL CONFIGURATION
# ==================================================

FINAL_TOP_K = 5
CANDIDATE_K = 40


# ==================================================
# 3. EVALUATE RETRIEVAL
# ==================================================

def evaluate_retrieval():

    print("\n========== RETRIEVAL EVALUATION ==========")

    for question_number, question in enumerate(
        test_questions,
        start=1
    ):

        print("\n" + "=" * 70)
        print(f"QUESTION {question_number}: {question}")
        print("=" * 70)

        # Retrieve and rerank chunks
        results = retrieve_with_rrf(
            queries=[question],
            top_k=FINAL_TOP_K,
            candidate_k=CANDIDATE_K,
            per_document=False,
            rerank=True
        )

        # Extract retrieval results
        ids = results["ids"][0]
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]
        scores = results["rerank_scores"][0]

        if not documents:
            print("No relevant chunks found.")
            continue

        # Display top retrieved chunks
        for rank, (
            chunk_id,
            document,
            metadata,
            distance,
            score
        ) in enumerate(
            zip(
                ids,
                documents,
                metadatas,
                distances,
                scores
            ),
            start=1
        ):

            print(f"\n--- RESULT {rank} ---")

            print("Chunk ID:", chunk_id)
            print("Source:", metadata["source"])
            print("Page:", metadata["page_number"])
            print("L2 Distance:", round(distance, 4))
            print("Rerank Score:", round(score, 4))

            print("\nRetrieved Content:")
            print(document)

        print("\nTotal retrieved chunks:", len(documents))


# ==================================================
# 4. RUN EVALUATION
# ==================================================

if __name__ == "__main__":
    evaluate_retrieval()
