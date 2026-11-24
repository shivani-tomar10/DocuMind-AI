
import json
from pathlib import Path


# ==================================================
# 1. LOAD RELEVANCE LABELS
# ==================================================

BASE_DIR = Path(__file__).resolve().parent

LABELS_FILE = BASE_DIR / "relevance_labels.json"

with open(LABELS_FILE, "r", encoding="utf-8") as file:
    relevance_labels = json.load(file)


# ==================================================
# 2. PRECISION@K
# ==================================================

def precision_at_k(retrieved_ids, labels, k=5):

    top_k_ids = retrieved_ids[:k]

    if not top_k_ids:
        return 0.0

    relevant_count = sum(
        1 for chunk_id in top_k_ids
        if labels.get(chunk_id, 0) > 0
    )

    return relevant_count / k


# ==================================================
# 3. RECALL@K
# ==================================================

def recall_at_k(retrieved_ids, labels, k=5):

    top_k_ids = retrieved_ids[:k]

    total_relevant = sum(
        1 for score in labels.values()
        if score > 0
    )

    if total_relevant == 0:
        return 0.0

    retrieved_relevant = sum(
        1 for chunk_id in top_k_ids
        if labels.get(chunk_id, 0) > 0
    )

    return retrieved_relevant / total_relevant


# ==================================================
# 4. NDCG@K
# ==================================================

def dcg_at_k(relevance_scores, k=5):

    dcg = 0.0

    for index, score in enumerate(relevance_scores[:k]):

        rank = index + 1

        dcg += score / (
            __import__("math").log2(rank + 1)
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


# ==================================================
# 5. CALCULATE METRICS
# ==================================================

def evaluate_metrics():

    print("\n========== RETRIEVAL METRICS ==========")

    precision_scores = []
    recall_scores = []
    ndcg_scores = []

    for question, labels in relevance_labels.items():

        retrieved_ids = list(labels.keys())

        precision = precision_at_k(
            retrieved_ids, labels, k=5
        )

        recall = recall_at_k(
            retrieved_ids, labels, k=5
        )

        ndcg = ndcg_at_k(
            retrieved_ids, labels, k=5
        )

        precision_scores.append(precision)
        recall_scores.append(recall)
        ndcg_scores.append(ndcg)

        print("\nQuestion:", question)
        print("Precision@5:", round(precision, 4))
        print("Recall@5:", round(recall, 4))
        print("NDCG@5:", round(ndcg, 4))

    print("\n========== AVERAGE METRICS ==========")

    print(
        "Average Precision@5:",
        round(sum(precision_scores) / len(precision_scores), 4)
    )

    print(
        "Average Recall@5:",
        round(sum(recall_scores) / len(recall_scores), 4)
    )

    print(
        "Average NDCG@5:",
        round(sum(ndcg_scores) / len(ndcg_scores), 4)
    )


# ==================================================
# 6. RUN EVALUATION
# ==================================================

if __name__ == "__main__":
    evaluate_metrics()
