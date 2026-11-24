
import json
from pathlib import Path

from backend.retrieval import retrieve_with_rrf


# 1. CONFIGURATION

BASE_DIR = Path(__file__).resolve().parent

OUTPUT_FILE = BASE_DIR / "retrieval_results.json"

FINAL_TOP_K = 5
CANDIDATE_K = 40

test_questions = [
    "What is Artificial Intelligence?",
    "What are the goals of Artificial Intelligence?",
    "Explain Strong AI versus Weak AI.",
    "What is the Turing Test?",
    "What are the major parts of AI?"
]


# 2. RUN RETRIEVAL

def run_evaluation():

    all_results = []

    for question_number, question in enumerate(
        test_questions,
        start=1
    ):

        print(f"\nQuestion {question_number}: {question}")

        results = retrieve_with_rrf(
            queries=[question],
            top_k=FINAL_TOP_K,
            candidate_k=CANDIDATE_K,
            per_document=False,
            rerank=True
        )

        ids = results["ids"][0]
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]
        scores = results["rerank_scores"][0]

        retrieved_chunks = []

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

            chunk = {
                "rank": rank,
                "chunk_id": chunk_id,
                "source": metadata["source"],
                "page_number": metadata["page_number"],
                "document": document,
                "distance": float(distance),
                "rerank_score": float(score)
            }

            retrieved_chunks.append(chunk)

        all_results.append({
            "question": question,
            "retrieved_chunks": retrieved_chunks
        })

        print("Retrieved chunks:", len(retrieved_chunks))

    # 3. SAVE RESULTS

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            all_results,
            file,
            indent=4,
            ensure_ascii=False
        )

    print("\n================================")
    print("RETRIEVAL EVALUATION COMPLETED")
    print("================================")
    print("Questions evaluated:", len(all_results))
    print("Saved to:", OUTPUT_FILE)


# 4. EXECUTION

if __name__ == "__main__":
    run_evaluation()
