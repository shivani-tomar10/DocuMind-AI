
import json
import math
from pathlib import Path

# --------------------------------------------------
# FILE PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

RESULTS_FILE = BASE_DIR / "expanded_retrieval_results.json"
LABELS_FILE = BASE_DIR / "expanded_relevance_labels.json"

REPORT_FILE = BASE_DIR / "expanded_metrics_report.json"

TOP_K = 5


# --------------------------------------------------
# LOAD JSON FILES
# --------------------------------------------------

with open(RESULTS_FILE, "r", encoding="utf-8") as file:
    retrieval_data = json.load(file)

with open(LABELS_FILE, "r", encoding="utf-8") as file:
    relevance_labels = json.load(file)


# --------------------------------------------------
# PRECISION@K
# --------------------------------------------------

def calculate_precision(retrieved_ids, labels, k=5):

    top_results = retrieved_ids[:k]

    if not top_results:
        return 0.0

    relevant_count = sum(
        1 for chunk_id in top_results
        if labels.get(chunk_id, 0) > 0
    )

    return relevant_count / k


# --------------------------------------------------
# NDCG@K
# --------------------------------------------------

def calculate_dcg(scores):

    dcg = 0.0

    for rank, score in enumerate(scores, start=1):

        gain = (2 ** score) - 1

        dcg += gain / math.log2(rank + 1)

    return dcg


def calculate_ndcg(retrieved_ids, labels, k=5):

    actual_scores = [
        labels.get(chunk_id, 0)
        for chunk_id in retrieved_ids[:k]
    ]

    ideal_scores = sorted(
        labels.values(),
        reverse=True
    )[:k]

    dcg = calculate_dcg(actual_scores)
    idcg = calculate_dcg(ideal_scores)

    if idcg == 0:
        return None

    return dcg / idcg


# --------------------------------------------------
# EVALUATE BENCHMARK
# --------------------------------------------------

print("\n========== EXPANDED RETRIEVAL EVALUATION ==========")

question_results = []

precision_scores = []
ndcg_scores = []

unanswerable_questions = []
unanswerable_false_positive_rates = []

for question in retrieval_data:

    question_id = question["id"]
    question_text = question["question"]

    retrieved_ids = [
        chunk["chunk_id"]
        for chunk in question["retrieved_chunks"]
    ]

    labels = relevance_labels.get(question_id, {})

    answerable = question["answerable"]

    precision = calculate_precision(
        retrieved_ids,
        labels,
        TOP_K
    )

    ndcg = calculate_ndcg(
        retrieved_ids,
        labels,
        TOP_K
    )

    # ----------------------------------------------
    # Answerable questions
    # ----------------------------------------------

    if answerable:

        precision_scores.append(precision)

        if ndcg is not None:
            ndcg_scores.append(ndcg)

    # ----------------------------------------------
    # Unanswerable questions
    # ----------------------------------------------

    else:

        false_positive_count = sum(
            1 for chunk_id in retrieved_ids[:TOP_K]
            if labels.get(chunk_id, 0) > 0
        )

        false_positive_rate = (
            false_positive_count / TOP_K
        )

        unanswerable_false_positive_rates.append(
            false_positive_rate
        )

        unanswerable_questions.append(question_id)

    # ----------------------------------------------
    # Print question results
    # ----------------------------------------------

    print(f"\n{question_id}: {question_text}")

    print(
        f"Precision@{TOP_K}: {precision:.4f}"
    )

    print(
        "NDCG@{}: {}".format(
            TOP_K,
            f"{ndcg:.4f}" if ndcg is not None else "N/A"
        )
    )

    if not answerable:

        print(
            f"Unanswerable retrieval false-positive rate: "
            f"{false_positive_rate:.4f}"
        )

    question_results.append({
        "id": question_id,
        "question": question_text,
        "answerable": answerable,
        "precision_at_5": precision,
        "ndcg_at_5": ndcg
    })


# --------------------------------------------------
# CALCULATE FINAL AVERAGES
# --------------------------------------------------

average_precision = (
    sum(precision_scores) / len(precision_scores)
    if precision_scores else 0.0
)

average_ndcg = (
    sum(ndcg_scores) / len(ndcg_scores)
    if ndcg_scores else 0.0
)

average_unanswerable_false_positive_rate = (
    sum(unanswerable_false_positive_rates)
    / len(unanswerable_false_positive_rates)
    if unanswerable_false_positive_rates else 0.0
)


# --------------------------------------------------
# FINAL REPORT
# --------------------------------------------------

report = {

    "total_questions": len(retrieval_data),

    "answerable_questions": len(precision_scores),

    "unanswerable_questions": len(unanswerable_questions),

    "average_precision_at_5": round(
        average_precision, 4
    ),

    "average_ndcg_at_5": round(
        average_ndcg, 4
    ),

    "average_unanswerable_retrieval_false_positive_rate": round(
        average_unanswerable_false_positive_rate, 4
    ),

    "recall_at_5": None,

    "recall_note": (
        "Recall@5 is not calculated because the complete set "
        "of relevant chunks in the corpus has not been judged."
    ),

    "question_results": question_results
}


# --------------------------------------------------
# SAVE REPORT
# --------------------------------------------------

with open(REPORT_FILE, "w", encoding="utf-8") as file:

    json.dump(
        report,
        file,
        indent=4
    )


# --------------------------------------------------
# DISPLAY SUMMARY
# --------------------------------------------------

print("\n========== FINAL EVALUATION SUMMARY ==========")

print("Total questions:", report["total_questions"])

print("Answerable questions:", report["answerable_questions"])

print("Unanswerable questions:", report["unanswerable_questions"])

print(
    "Average Precision@5:",
    f"{average_precision:.4f}"
)

print(
    "Average NDCG@5:",
    f"{average_ndcg:.4f}"
)

print(
    "Average unanswerable retrieval false-positive rate:",
    f"{average_unanswerable_false_positive_rate:.4f}"
)

print("Recall@5: Not calculated")

print("\nReport saved to:", REPORT_FILE)

print("\n========== EVALUATION COMPLETED ==========")
