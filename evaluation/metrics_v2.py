
import json
import math
from pathlib import Path


# ==========================================
# 1. FILE PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

RESULTS_FILE = BASE_DIR / "retrieval_results.json"
LABELS_FILE = BASE_DIR / "relevance_labels.json"
REPORT_FILE = BASE_DIR / "metrics_report.json"


# ==========================================
# 2. LOAD JSON FILES
# ==========================================

def load_json(file_path):

    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


retrieval_results = load_json(RESULTS_FILE)
relevance_labels = load_json(LABELS_FILE)


# ==========================================
# 3. PRECISION@K
# ==========================================

def precision_at_k(retrieved_ids, labels, k=5):

    top_k = retrieved_ids[:k]

    if not top_k:
        return 0.0

    relevant_count = sum(
        1 for chunk_id in top_k
        if labels.get(chunk_id, 0) > 0
    )

    return relevant_count / k


# ==========================================
# 4. POOLED RECALL@K
# ==========================================

def recall_at_k(retrieved_ids, labels, k=5):

    total_relevant = sum(
        1 for score in labels.values()
        if score > 0
    )

    if total_relevant == 0:
        return 0.0

    retrieved_relevant = sum(
        1 for chunk_id in retrieved_ids[:k]
        if labels.get(chunk_id, 0) > 0
    )

    return retrieved_relevant / total_relevant


# ==========================================
# 5. DCG AND NDCG@K
# ==========================================

def dcg_at_k(scores, k=5):

    dcg = 0.0

    for rank, score in enumerate(scores[:k], start=1):

        dcg += (
            (2 ** score - 1)
            / math.log2(rank + 1)
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
# 6. EVALUATE ALL QUESTIONS
# ==========================================

def evaluate():

    report = []

    precision_values = []
    recall_values = []
    ndcg_values = []

    print("\n========== RETRIEVAL METRICS V2 ==========")

    for item in retrieval_results:

        question = item["question"]

        retrieved_ids = [
            chunk["chunk_id"]
            for chunk in item["retrieved_chunks"]
        ]

        labels = relevance_labels.get(question)

        if labels is None:
            raise ValueError(
                f"Missing relevance labels for: {question}"
            )

        precision = precision_at_k(
            retrieved_ids, labels, k=5
        )

        recall = recall_at_k(
            retrieved_ids, labels, k=5
        )

        ndcg = ndcg_at_k(
            retrieved_ids, labels, k=5
        )

        precision_values.append(precision)
        recall_values.append(recall)
        ndcg_values.append(ndcg)

        question_report = {
            "question": question,
            "precision_at_5": precision,
            "pooled_recall_at_5": recall,
            "ndcg_at_5": ndcg
        }

        report.append(question_report)

        print("\nQuestion:", question)
        print("Precision@5:", round(precision, 4))
        print("Pooled Recall@5:", round(recall, 4))
        print("NDCG@5:", round(ndcg, 4))

    # ======================================
    # 7. AVERAGE METRICS
    # ======================================

    averages = {
        "average_precision_at_5": (
            sum(precision_values) / len(precision_values)
        ),
        "average_pooled_recall_at_5": (
            sum(recall_values) / len(recall_values)
        ),
        "average_ndcg_at_5": (
            sum(ndcg_values) / len(ndcg_values)
        )
    }

    print("\n========== AVERAGE METRICS ==========")

    for metric, value in averages.items():
        print(metric, ":", round(value, 4))

    # ======================================
    # 8. SAVE REPORT
    # ======================================

    final_report = {
        "evaluation_questions": len(report),
        "results": report,
        "averages": averages
    }

    with open(
        REPORT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            final_report,
            file,
            indent=4,
            ensure_ascii=False
        )

    print("\nReport saved to:", REPORT_FILE)


# ==========================================
# 9. EXECUTION
# ==========================================

if __name__ == "__main__":
    evaluate()
