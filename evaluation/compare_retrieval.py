
import json
import math
from pathlib import Path

from backend.retrieval import retrieve_with_rrf


# ==================================================
# 1. CONFIGURATION
# ==================================================

BASE_DIR = Path(__file__).resolve().parent

QUESTIONS_FILE = BASE_DIR / "retrieval_questions.json"
LABELS_FILE = BASE_DIR / "relevance_labels.json"
OUTPUT_FILE = BASE_DIR / "retrieval_comparison.json"

EVALUATION_SOURCE = "sample.pdf"

FINAL_TOP_K = 5
CANDIDATE_K = 40
RRF_K = 60

SEARCH_MODE = "exact"

TEST_QUESTION_IDS = ["Q01", "Q02", "Q03", "Q04", "Q05"]


# ==================================================
# 2. LOAD EVALUATION DATA
# ==================================================

def load_evaluation_data():

    if not QUESTIONS_FILE.exists():
        raise FileNotFoundError(
            f"Questions file not found: {QUESTIONS_FILE}"
        )

    if not LABELS_FILE.exists():
        raise FileNotFoundError(
            f"Relevance labels not found: {LABELS_FILE}"
        )

    with open(QUESTIONS_FILE, "r", encoding="utf-8") as file:
        question_data = json.load(file)

    with open(LABELS_FILE, "r", encoding="utf-8") as file:
        relevance_labels = json.load(file)

    if not isinstance(question_data, list):
        raise ValueError(
            "Retrieval questions must be a JSON list."
        )

    if not isinstance(relevance_labels, dict):
        raise ValueError(
            "Relevance labels must be a JSON object."
        )

    question_map = {
        item["id"]: item
        for item in question_data
    }

    evaluation_questions = []

    for question_id in TEST_QUESTION_IDS:

        if question_id not in question_map:
            print(f"Warning: Question {question_id} not found.")
            continue

        if question_id not in relevance_labels:
            print(f"Warning: Labels for {question_id} not found.")
            continue

        item = question_map[question_id]

        evaluation_questions.append({
            "id": question_id,
            "question": item["question"],
            "labels": relevance_labels[question_id]
        })

    if not evaluation_questions:
        raise ValueError(
            "No matching evaluation questions and relevance labels found."
        )

    return evaluation_questions


# ==================================================
# 3. PRECISION@K
# ==================================================

def precision_at_k(retrieved_ids, labels, k=5):

    if k <= 0:
        return 0.0

    relevant = sum(
        1
        for chunk_id in retrieved_ids[:k]
        if labels.get(chunk_id, 0) > 0
    )

    return relevant / k


# ==================================================
# 4. RECALL@K
# ==================================================

def recall_at_k(retrieved_ids, labels, k=5):

    relevant_ids = {
        chunk_id
        for chunk_id, score in labels.items()
        if score > 0
    }

    if not relevant_ids:
        return 0.0

    retrieved_relevant = sum(
        1
        for chunk_id in retrieved_ids[:k]
        if chunk_id in relevant_ids
    )

    return retrieved_relevant / len(relevant_ids)


# ==================================================
# 5. DCG@K AND NDCG@K
# ==================================================

def dcg_at_k(scores, k=5):

    dcg = 0.0

    for rank, score in enumerate(scores[:k], start=1):

        gain = (2 ** score) - 1

        dcg += gain / math.log2(rank + 1)

    return dcg


def ndcg_at_k(retrieved_ids, labels, k=5):

    actual_scores = [
        labels.get(chunk_id, 0)
        for chunk_id in retrieved_ids[:k]
    ]

    ideal_scores = sorted(
        labels.values(),
        reverse=True
    )[:k]

    actual_dcg = dcg_at_k(actual_scores, k)
    ideal_dcg = dcg_at_k(ideal_scores, k)

    if ideal_dcg == 0:
        return 0.0

    return actual_dcg / ideal_dcg


# ==================================================
# 6. RUN ONE RETRIEVAL METHOD
# ==================================================

def run_retrieval(question, use_reranker):

    results = retrieve_with_rrf(
        queries=[question],
        top_k=FINAL_TOP_K,
        candidate_k=CANDIDATE_K,
        rrf_k=RRF_K,
        per_document=False,
        rerank=use_reranker,
        source=EVALUATION_SOURCE,
        search_mode=SEARCH_MODE
    )

    retrieved_chunks = []

    for rank, (
        chunk_id,
        document,
        metadata,
        distance
    ) in enumerate(
        zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        ),
        start=1
    ):

        chunk = {
            "rank": rank,
            "chunk_id": chunk_id,
            "source": metadata["source"],
            "page_number": metadata["page_number"],
            "document": document,
            "distance": float(distance)
        }

        if use_reranker:
            chunk["rerank_score"] = float(
                results["rerank_scores"][0][rank - 1]
            )

        retrieved_chunks.append(chunk)

    return retrieved_chunks


# ==================================================
# 7. EVALUATE BOTH METHODS
# ==================================================

def evaluate_comparison():

    evaluation_questions = load_evaluation_data()

    comparison_results = []

    baseline_precision = []
    reranked_precision = []

    baseline_recall = []
    reranked_recall = []

    baseline_ndcg = []
    reranked_ndcg = []

    print("\n========== RETRIEVAL A/B COMPARISON ==========")

    print("Evaluation source:", EVALUATION_SOURCE)
    print("Search mode:", SEARCH_MODE)
    print("Questions:", len(evaluation_questions))

    for item in evaluation_questions:

        question_id = item["id"]
        question = item["question"]
        labels = item["labels"]

        print("\n" + "=" * 60)
        print("Question ID:", question_id)
        print("Question:", question)
        print("=" * 60)

        # Method A: RRF without Cross-Encoder.
        baseline_chunks = run_retrieval(
            question,
            use_reranker=False
        )

        # Method B: RRF with Cross-Encoder.
        reranked_chunks = run_retrieval(
            question,
            use_reranker=True
        )

        baseline_ids = [
            chunk["chunk_id"]
            for chunk in baseline_chunks
        ]

        reranked_ids = [
            chunk["chunk_id"]
            for chunk in reranked_chunks
        ]

        # Calculate baseline metrics.
        p_baseline = precision_at_k(
            baseline_ids, labels, FINAL_TOP_K
        )

        r_baseline = recall_at_k(
            baseline_ids, labels, FINAL_TOP_K
        )

        n_baseline = ndcg_at_k(
            baseline_ids, labels, FINAL_TOP_K
        )

        # Calculate reranked metrics.
        p_reranked = precision_at_k(
            reranked_ids, labels, FINAL_TOP_K
        )

        r_reranked = recall_at_k(
            reranked_ids, labels, FINAL_TOP_K
        )

        n_reranked = ndcg_at_k(
            reranked_ids, labels, FINAL_TOP_K
        )

        baseline_precision.append(p_baseline)
        reranked_precision.append(p_reranked)

        baseline_recall.append(r_baseline)
        reranked_recall.append(r_reranked)

        baseline_ndcg.append(n_baseline)
        reranked_ndcg.append(n_reranked)

        print("\nBaseline RRF:")
        print("Retrieved IDs:", baseline_ids)
        print("Precision@5:", round(p_baseline, 4))
        print("Recall@5:", round(r_baseline, 4))
        print("NDCG@5:", round(n_baseline, 4))

        print("\nRRF + Cross-Encoder:")
        print("Retrieved IDs:", reranked_ids)
        print("Precision@5:", round(p_reranked, 4))
        print("Recall@5:", round(r_reranked, 4))
        print("NDCG@5:", round(n_reranked, 4))

        comparison_results.append({
            "question_id": question_id,
            "question": question,
            "source": EVALUATION_SOURCE,

            "baseline": {
                "retrieved_chunks": baseline_chunks,
                "precision_at_5": p_baseline,
                "recall_at_5": r_baseline,
                "ndcg_at_5": n_baseline
            },

            "reranked": {
                "retrieved_chunks": reranked_chunks,
                "precision_at_5": p_reranked,
                "recall_at_5": r_reranked,
                "ndcg_at_5": n_reranked
            }
        })

    # ==================================================
    # 8. AVERAGE COMPARISON
    # ==================================================

    count = len(evaluation_questions)

    averages = {
        "baseline": {
            "precision_at_5": sum(baseline_precision) / count,
            "recall_at_5": sum(baseline_recall) / count,
            "ndcg_at_5": sum(baseline_ndcg) / count
        },

        "reranked": {
            "precision_at_5": sum(reranked_precision) / count,
            "recall_at_5": sum(reranked_recall) / count,
            "ndcg_at_5": sum(reranked_ndcg) / count
        }
    }

    print("\n========== AVERAGE COMPARISON ==========")

    for method, metrics in averages.items():

        print(f"\n{method.upper()}")

        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")

    # ==================================================
    # 9. SAVE COMPARISON REPORT
    # ==================================================

    final_report = {
        "questions_evaluated": count,
        "question_ids": [
            item["id"] for item in evaluation_questions
        ],
        "evaluation_source": EVALUATION_SOURCE,
        "search_mode": SEARCH_MODE,
        "candidate_k": CANDIDATE_K,
        "final_top_k": FINAL_TOP_K,
        "averages": averages,
        "results": comparison_results
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:

        json.dump(
            final_report,
            file,
            indent=4,
            ensure_ascii=False
        )

    print("\nComparison report saved to:")
    print(OUTPUT_FILE)


# ==================================================
# 10. EXECUTION
# ==================================================

if __name__ == "__main__":
    evaluate_comparison()
