
import json
from pathlib import Path

from backend.retrieval import retrieve_with_rrf


# ==========================================
# 1. CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

LABELS_FILE = BASE_DIR / "relevance_labels.json"
OUTPUT_FILE = BASE_DIR / "retrieval_comparison.json"

FINAL_TOP_K = 5
CANDIDATE_K = 40

test_questions = [
    "What is Artificial Intelligence?",
    "What are the goals of Artificial Intelligence?",
    "Explain Strong AI versus Weak AI.",
    "What is the Turing Test?",
    "What are the major parts of AI?"
]


# ==========================================
# 2. LOAD RELEVANCE LABELS
# ==========================================

with open(
    LABELS_FILE,
    "r",
    encoding="utf-8"
) as file:

    relevance_labels = json.load(file)


# ==========================================
# 3. CALCULATE PRECISION@5
# ==========================================

def precision_at_k(retrieved_ids, labels, k=5):

    top_k = retrieved_ids[:k]

    if not top_k:
        return 0.0

    relevant = sum(
        1 for chunk_id in top_k
        if labels.get(chunk_id, 0) > 0
    )

    return relevant / k


# ==========================================
# 4. CALCULATE NDCG@5
# ==========================================

def dcg_at_k(scores, k=5):

    dcg = 0.0

    for rank, score in enumerate(scores[:k], start=1):

        dcg += (
            (2 ** score - 1)
            / __import__("math").log2(rank + 1)
        )

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


# ==========================================
# 5. RUN ONE RETRIEVAL METHOD
# ==========================================

def run_retrieval(question, use_reranker):

    results = retrieve_with_rrf(
        queries=[question],
        top_k=FINAL_TOP_K,
        candidate_k=CANDIDATE_K,
        per_document=False,
        rerank=use_reranker
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


# ==========================================
# 6. EVALUATE BOTH METHODS
# ==========================================

def evaluate_comparison():

    comparison_results = []

    baseline_precision = []
    reranked_precision = []

    baseline_ndcg = []
    reranked_ndcg = []

    print("\n========== RETRIEVAL A/B COMPARISON ==========")

    for question in test_questions:

        print("\nQuestion:", question)

        labels = relevance_labels[question]

        # Method A: RRF without cross-encoder
        baseline_chunks = run_retrieval(
            question,
            use_reranker=False
        )

        # Method B: RRF with cross-encoder
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

        # Calculate metrics
        p_baseline = precision_at_k(
            baseline_ids, labels
        )

        p_reranked = precision_at_k(
            reranked_ids, labels
        )

        n_baseline = ndcg_at_k(
            baseline_ids, labels
        )

        n_reranked = ndcg_at_k(
            reranked_ids, labels
        )

        baseline_precision.append(p_baseline)
        reranked_precision.append(p_reranked)

        baseline_ndcg.append(n_baseline)
        reranked_ndcg.append(n_reranked)

        print("\nBaseline RRF:")
        print("Precision@5:", round(p_baseline, 4))
        print("NDCG@5:", round(n_baseline, 4))

        print("\nRRF + Cross-Encoder:")
        print("Precision@5:", round(p_reranked, 4))
        print("NDCG@5:", round(n_reranked, 4))

        comparison_results.append({
            "question": question,

            "baseline": {
                "retrieved_chunks": baseline_chunks,
                "precision_at_5": p_baseline,
                "ndcg_at_5": n_baseline
            },

            "reranked": {
                "retrieved_chunks": reranked_chunks,
                "precision_at_5": p_reranked,
                "ndcg_at_5": n_reranked
            }
        })

    # ======================================
    # 7. AVERAGE COMPARISON
    # ======================================

    averages = {
        "baseline": {
            "precision_at_5": (
                sum(baseline_precision)
                / len(baseline_precision)
            ),
            "ndcg_at_5": (
                sum(baseline_ndcg)
                / len(baseline_ndcg)
            )
        },

        "reranked": {
            "precision_at_5": (
                sum(reranked_precision)
                / len(reranked_precision)
            ),
            "ndcg_at_5": (
                sum(reranked_ndcg)
                / len(reranked_ndcg)
            )
        }
    }

    print("\n========== AVERAGE COMPARISON ==========")

    print("\nBaseline RRF:")
    print(
        "Precision@5:",
        round(averages["baseline"]["precision_at_5"], 4)
    )
    print(
        "NDCG@5:",
        round(averages["baseline"]["ndcg_at_5"], 4)
    )

    print("\nRRF + Cross-Encoder:")
    print(
        "Precision@5:",
        round(averages["reranked"]["precision_at_5"], 4)
    )
    print(
        "NDCG@5:",
        round(averages["reranked"]["ndcg_at_5"], 4)
    )

    # ======================================
    # 8. SAVE COMPARISON REPORT
    # ======================================

    final_report = {
        "questions_evaluated": len(comparison_results),
        "candidate_k": CANDIDATE_K,
        "final_top_k": FINAL_TOP_K,
        "averages": averages,
        "results": comparison_results
    }

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            final_report,
            file,
            indent=4,
            ensure_ascii=False
        )

    print("\nComparison report saved to:")
    print(OUTPUT_FILE)


# ==========================================
# 9. EXECUTION
# ==========================================

if __name__ == "__main__":
    evaluate_comparison()
