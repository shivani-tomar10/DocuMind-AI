
import json
import math
from pathlib import Path

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

RESULTS_FILE = BASE_DIR / "expanded_retrieval_results.json"
LABELS_FILE = BASE_DIR / "expanded_relevance_labels.json"
REPORT_FILE = BASE_DIR / "expanded_metrics_report.json"

TOP_K = 5


# --------------------------------------------------
# LOAD AND VALIDATE JSON FILES
# --------------------------------------------------

def load_json(file_path):
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


retrieval_data = load_json(RESULTS_FILE)
relevance_labels = load_json(LABELS_FILE)

if not isinstance(retrieval_data, list):
    raise ValueError("Retrieval results must be a JSON list.")

if not isinstance(relevance_labels, dict):
    raise ValueError("Relevance labels must be a JSON object.")


# --------------------------------------------------
# PRECISION@K
# --------------------------------------------------

def calculate_precision(retrieved_ids, labels, k=5):
    top_results = retrieved_ids[:k]

    if not top_results:
        return 0.0

    relevant_count = sum(
        1
        for chunk_id in top_results
        if labels.get(chunk_id, 0) > 0
    )

    # Fixed denominator: missing results count as misses.
    return relevant_count / k


# --------------------------------------------------
# DCG AND NDCG
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
# UNANSWERABLE QUESTION METRIC
# --------------------------------------------------

def calculate_non_relevant_rate(retrieved_ids, labels, k=5):
    """
    Measures the proportion of judged retrieved chunks
    labeled as non-relevant for an unanswerable question.

    Unjudged chunks are excluded from the calculation.
    """

    top_results = retrieved_ids[:k]

    judged_chunks = [
        chunk_id
        for chunk_id in top_results
        if chunk_id in labels
    ]

    if not judged_chunks:
        return None, 0, 0

    non_relevant_count = sum(
        1
        for chunk_id in judged_chunks
        if labels[chunk_id] == 0
    )

    rate = non_relevant_count / len(judged_chunks)

    return rate, non_relevant_count, len(judged_chunks)


# --------------------------------------------------
# EVALUATE BENCHMARK
# --------------------------------------------------

print("\n========== EXPANDED RETRIEVAL EVALUATION ==========")

question_results = []

precision_scores = []
ndcg_scores = []
unanswerable_rates = []

unanswerable_questions = []

for question in retrieval_data:

    question_id = question["id"]
    question_text = question["question"]

    if "answerable" not in question:
        raise ValueError(
            f"Missing answerable field for {question_id}"
        )

    retrieved_chunks = question.get("retrieved_chunks", [])

    retrieved_ids = [
        chunk["chunk_id"]
        for chunk in retrieved_chunks
    ]

    labels = relevance_labels.get(question_id, {})

    answerable = question["answerable"]

    if not isinstance(labels, dict):
        raise ValueError(
            f"Invalid relevance labels for {question_id}"
        )

    # ----------------------------------------------
    # Answerable questions
    # ----------------------------------------------

    if answerable:

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

        precision_scores.append(precision)

        if ndcg is not None:
            ndcg_scores.append(ndcg)

        print(f"\n{question_id}: {question_text}")
        print(f"Precision@{TOP_K}: {precision:.4f}")

        print(
            f"NDCG@{TOP_K}: "
            f"{ndcg:.4f}" if ndcg is not None
            else f"NDCG@{TOP_K}: N/A"
        )

        question_results.append({
            "id": question_id,
            "question": question_text,
            "answerable": True,
            "precision_at_5": round(precision, 4),
            "ndcg_at_5": (
                round(ndcg, 4)
                if ndcg is not None else None
            )
        })

    # ----------------------------------------------
    # Unanswerable questions
    # ----------------------------------------------

    else:

        rate, non_relevant_count, judged_count = (
            calculate_non_relevant_rate(
                retrieved_ids,
                labels,
                TOP_K
            )
        )

        unanswerable_questions.append(question_id)

        if rate is not None:
            unanswerable_rates.append(rate)

        unjudged_count = min(TOP_K, len(retrieved_ids)) - judged_count

        print(f"\n{question_id}: {question_text}")
        print("Question type: Unanswerable")

        print(
            f"Non-relevant retrieval rate: "
            f"{rate:.4f}" if rate is not None
            else "Non-relevant retrieval rate: N/A"
        )

        print(f"Non-relevant chunks: {non_relevant_count}")
        print(f"Judged chunks: {judged_count}")
        print(f"Unjudged chunks: {unjudged_count}")

        question_results.append({
            "id": question_id,
            "question": question_text,
            "answerable": False,
            "non_relevant_retrieval_rate": (
                round(rate, 4)
                if rate is not None else None
            ),
            "non_relevant_chunks": non_relevant_count,
            "judged_chunks": judged_count,
            "unjudged_chunks": unjudged_count
        })


# --------------------------------------------------
# CALCULATE FINAL AVERAGES
# --------------------------------------------------

average_precision = (
    sum(precision_scores) / len(precision_scores)
    if precision_scores else None
)

average_ndcg = (
    sum(ndcg_scores) / len(ndcg_scores)
    if ndcg_scores else None
)

average_non_relevant_rate = (
    sum(unanswerable_rates) / len(unanswerable_rates)
    if unanswerable_rates else None
)


# --------------------------------------------------
# FINAL REPORT
# --------------------------------------------------

report = {
    "total_questions": len(retrieval_data),

    "top_k": TOP_K,

    "answerable_questions": len(precision_scores),

    "unanswerable_questions": len(unanswerable_questions),

    "average_precision_at_5": (
        round(average_precision, 4)
        if average_precision is not None else None
    ),

    "average_ndcg_at_5": (
        round(average_ndcg, 4)
        if average_ndcg is not None else None
    ),

    "average_unanswerable_non_relevant_retrieval_rate": (
        round(average_non_relevant_rate, 4)
        if average_non_relevant_rate is not None else None
    ),

    "ndcg_method": "Exponential gain: (2^relevance - 1)",

    "unanswerable_metric_note": (
        "Measures the proportion of judged retrieved chunks "
        "labeled non-relevant (grade 0) for unanswerable questions. "
        "Unjudged chunks are excluded."
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
        indent=4,
        ensure_ascii=False
    )


# --------------------------------------------------
# FINAL SUMMARY
# --------------------------------------------------

print("\n========== FINAL SUMMARY ==========")

print(f"Total questions: {report['total_questions']}")
print(f"Answerable questions: {report['answerable_questions']}")
print(f"Unanswerable questions: {report['unanswerable_questions']}")

print(f"Average Precision@5: {average_precision}")
print(f"Average NDCG@5: {average_ndcg}")

print(
    "Average unanswerable non-relevant retrieval rate:",
    average_non_relevant_rate
)

print(f"\nReport saved to: {REPORT_FILE}")
