
from backend.retrieval import retrieve_with_rrf
import math


# ==================================================
# 1. CONFIGURATION
# ==================================================

EVALUATION_SOURCE = "sample.pdf"

TOP_K = 5
CANDIDATE_K = 40
RRF_K = 60

# Relevance scores:
# 2 = Highly relevant
# 1 = Partially relevant
# 0 = Not relevant

RELEVANCE_LABELS = {
    "What is Artificial Intelligence?": {
        "sample_p23_chunk3": 2,
        "sample_p21_chunk1": 1,
        "sample_p23_chunk1": 2,
        "sample_p20_chunk3": 1,
        "sample_p8_chunk1": 1
    },

    "What are the goals of Artificial Intelligence?": {
        "sample_p23_chunk1": 2,
        "sample_p20_chunk3": 1,
        "sample_p33_chunk3": 1,
        "sample_p2_chunk1": 0,
        "sample_p21_chunk1": 0
    },

    "Explain Strong AI versus Weak AI.": {
        "sample_p23_chunk4": 2,
        "sample_p23_chunk6": 2,
        "sample_p24_chunk1": 2,
        "sample_p41_chunk2": 1,
        "sample_p20_chunk3": 1
    },

    "What is the Turing Test?": {
        "sample_p24_chunk3": 2,
        "sample_p25_chunk1": 1,
        "sample_p20_chunk3": 1,
        "sample_p24_chunk2": 1,
        "sample_p24_chunk5": 2
    },

    "What are the major parts of AI?": {
        "sample_p37_chunk2": 2,
        "sample_p8_chunk2": 2,
        "sample_p41_chunk3": 2,
        "sample_p20_chunk3": 1,
        "sample_p37_chunk4": 1
    }
}


# ==================================================
# 2. PRECISION@K
# ==================================================

def precision_at_k(retrieved_ids, relevance, k=5):

    if k <= 0:
        return 0.0

    relevant_count = sum(
        1
        for chunk_id in retrieved_ids[:k]
        if relevance.get(chunk_id, 0) > 0
    )

    return relevant_count / k


# ==================================================
# 3. RECALL@K
# ==================================================

def recall_at_k(retrieved_ids, relevance, k=5):

    relevant_ids = {
        chunk_id
        for chunk_id, score in relevance.items()
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
# 4. NDCG@K
# ==================================================

def ndcg_at_k(retrieved_ids, relevance, k=5):

    dcg = 0.0

    for rank, chunk_id in enumerate(retrieved_ids[:k], start=1):

        relevance_score = relevance.get(chunk_id, 0)

        dcg += relevance_score / math.log2(rank + 1)

    ideal_scores = sorted(
        relevance.values(),
        reverse=True
    )[:k]

    idcg = sum(
        score / math.log2(rank + 1)
        for rank, score in enumerate(ideal_scores, start=1)
    )

    if idcg == 0:
        return 0.0

    return dcg / idcg


# ==================================================
# 5. EVALUATE RETRIEVAL
# ==================================================

def evaluate_retrieval():

    precision_scores = []
    recall_scores = []
    ndcg_scores = []

    print("=" * 65)
    print("DOCUMIND AI - RETRIEVAL METRICS EVALUATION")
    print("=" * 65)

    print("Evaluation PDF:", EVALUATION_SOURCE)
    print("Search mode: Exact NumPy L2")
    print("Top K:", TOP_K)
    print("Candidate K:", CANDIDATE_K)

    for question, relevance in RELEVANCE_LABELS.items():

        print("\n" + "=" * 65)
        print("QUESTION:", question)
        print("=" * 65)

        # Retrieve actual chunks using exact search.
        results = retrieve_with_rrf(
            queries=[question],
            top_k=TOP_K,
            candidate_k=CANDIDATE_K,
            rrf_k=RRF_K,
            per_document=False,
            rerank=True,
            source=EVALUATION_SOURCE,
            search_mode="exact"
        )

        retrieved_ids = results["ids"][0]

        print("\nRetrieved count:", len(retrieved_ids))
        print("Retrieved IDs:", retrieved_ids)

        if not retrieved_ids:
            print("WARNING: No chunks retrieved.")

        # Calculate metrics.
        precision = precision_at_k(
            retrieved_ids,
            relevance,
            TOP_K
        )

        recall = recall_at_k(
            retrieved_ids,
            relevance,
            TOP_K
        )

        ndcg = ndcg_at_k(
            retrieved_ids,
            relevance,
            TOP_K
        )

        precision_scores.append(precision)
        recall_scores.append(recall)
        ndcg_scores.append(ndcg)

        print("\nPrecision@5:", round(precision, 4))
        print("Recall@5:", round(recall, 4))
        print("NDCG@5:", round(ndcg, 4))

    # ==================================================
    # 6. FINAL AVERAGE METRICS
    # ==================================================

    total_questions = len(RELEVANCE_LABELS)

    if total_questions == 0:
        print("\nNo evaluation questions available.")
        return

    average_precision = sum(precision_scores) / total_questions
    average_recall = sum(recall_scores) / total_questions
    average_ndcg = sum(ndcg_scores) / total_questions

    print("\n" + "=" * 65)
    print("FINAL AVERAGE RETRIEVAL METRICS")
    print("=" * 65)

    print("Total questions:", total_questions)

    print("Average Precision@5:",
          round(average_precision, 4))

    print("Average Recall@5:",
          round(average_recall, 4))

    print("Average NDCG@5:",
          round(average_ndcg, 4))

    print("=" * 65)


# ==================================================
# 7. MAIN
# ==================================================

if __name__ == "__main__":
    evaluate_retrieval()
