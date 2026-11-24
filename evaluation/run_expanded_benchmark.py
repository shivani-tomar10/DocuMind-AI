
import json
from pathlib import Path

from backend.retrieval import retrieve_with_rrf


# ==========================================
# 1. FILE CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

QUESTIONS_FILE = BASE_DIR / "retrieval_questions.json"
OUTPUT_FILE = BASE_DIR / "expanded_retrieval_results.json"

CANDIDATE_K = 40
FINAL_TOP_K = 5


# ==========================================
# 2. LOAD QUESTIONS
# ==========================================

def load_questions():

    with open(
        QUESTIONS_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


# ==========================================
# 3. RUN RETRIEVAL
# ==========================================

def run_benchmark():

    questions = load_questions()

    all_results = []

    print("\n========== EXPANDED RETRIEVAL BENCHMARK ==========")

    for item in questions:

        question_id = item["id"]
        question = item["question"]

        print(f"\n{question_id}: {question}")

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

            retrieved_chunks.append({

                "rank": rank,
                "chunk_id": chunk_id,
                "source": metadata["source"],
                "page_number": metadata["page_number"],
                "document": document,
                "distance": float(distance),
                "rerank_score": float(score)

            })

        question_result = {

            "id": question_id,
            "question": question,
            "expected_sources": item["expected_sources"],
            "answerable": item["answerable"],
            "retrieved_chunks": retrieved_chunks

        }

        all_results.append(question_result)

        print("Retrieved chunks:", len(retrieved_chunks))

    # ======================================
    # 4. SAVE RESULTS
    # ======================================

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

    print("\n========== BENCHMARK COMPLETED ==========")

    print("Questions processed:", len(all_results))

    print(
        "Total retrieved chunks:",
        sum(
            len(item["retrieved_chunks"])
            for item in all_results
        )
    )

    print("Results saved to:", OUTPUT_FILE)


# ==========================================
# 5. EXECUTION
# ==========================================

if __name__ == "__main__":
    run_benchmark()
