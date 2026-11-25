
import json
import math
from pathlib import Path

from backend.retrieval import retrieve_with_rrf
from backend.llm import generate_search_queries


# ==================================================
# 1. CONFIGURATION
# ==================================================

BASE_DIR = Path(__file__).resolve().parent

QUESTIONS_FILE = BASE_DIR / "retrieval_questions.json"
LABELS_FILE = BASE_DIR / "expanded_relevance_labels.json"

OUTPUT_FILE = BASE_DIR / "expanded_strategy_comparison.json"

CANDIDATE_K = 40
TOP_K = 5
RRF_K = 60


# ==================================================
# 2. LOAD DATA
# ==================================================

def load_json(file_path):

    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


# ==================================================
# 3. SOURCE MATCHING
# ==================================================

def source_matches(actual_source, expected_sources):

    actual = Path(actual_source).stem.lower()

    return any(
        actual == Path(source).stem.lower()
        or actual in Path(source).stem.lower()
        or Path(source).stem.lower() in actual
        for source in expected_sources
    )


# ==================================================
# 4. GENERATE ALTERNATIVE QUERIES
# ==================================================

def build_queries(question):

    alternatives = generate_search_queries(
        question,
        num_queries=3
    )

    queries = [question]

    for alternative in alternatives:

        cleaned = alternative.strip()

        if (
            cleaned
            and cleaned.lower()
            not in {
                query.lower()
                for query in queries
            }
        ):
            queries.append(cleaned)

    return queries


# ==================================================
# 5. RETRIEVE RESULTS
# ==================================================

def retrieve(question, queries, expected_sources):

    # Restrict retrieval to expected sources when available.
    # For unanswerable questions, search the full collection.
    results = retrieve_with_rrf(
        queries=queries,
        top_k=TOP_K,
        candidate_k=CANDIDATE_K,
        rrf_k=RRF_K,
        per_document=False,
        rerank=True,
        search_mode="exact"
    )

    retrieved_chunks = []

    for rank, (
        chunk_id,
        document,
        metadata,
        distance,
        score
    ) in enumerate(
        zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
            results["rerank_scores"][0]
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

    return retrieved_chunks


# ==================================================
# 6. EVALUATION METRICS
# ==================================================

def precision_at_k(retrieved_ids, relevance):

    relevant_count = sum(
        1
        for chunk_id in retrieved_ids[:TOP_K]
        if relevance.get(chunk_id, 0) > 0
    )

    return relevant_count / TOP_K


def recall_at_k(retrieved_ids, relevance):

    relevant_ids = {
        chunk_id
        for chunk_id, score in relevance.items()
        if score > 0
    }

    if not relevant_ids:
        return None

    retrieved_relevant = sum(
        1
        for chunk_id in retrieved_ids[:TOP_K]
        if chunk_id in relevant_ids
    )

    return retrieved_relevant / len(relevant_ids)


def ndcg_at_k(retrieved_ids, relevance):

    dcg = 0.0

    for rank, chunk_id in enumerate(
        retrieved_ids[:TOP_K],
        start=1
    ):

        score = relevance.get(chunk_id, 0)

        dcg += (
            (2 ** score - 1)
            / math.log2(rank + 1)
        )

    ideal_scores = sorted(
        relevance.values(),
        reverse=True
    )[:TOP_K]

    idcg = sum(
        (2 ** score - 1)
        / math.log2(rank + 1)
        for rank, score in enumerate(
            ideal_scores,
            start=1
        )
    )

    return dcg / idcg if idcg > 0 else 0.0


def unanswerable_non_relevant_rate(
    retrieved_ids,
    relevance
):

    judged_ids = [
        chunk_id
        for chunk_id in retrieved_ids
        if chunk_id in relevance
    ]

    if not judged_ids:
        return None

    non_relevant = sum(
        1
        for chunk_id in judged_ids
        if relevance[chunk_id] == 0
    )

    return non_relevant / len(judged_ids)


# ==================================================
# 7. CALCULATE QUESTION METRICS
# ==================================================

def calculate_metrics(
    retrieved_chunks,
    relevance,
    answerable
):

    retrieved_ids = [
        chunk["chunk_id"]
        for chunk in retrieved_chunks
    ]

    if answerable:

        return {
            "precision_at_5": precision_at_k(
                retrieved_ids,
                relevance
            ),
            "recall_at_5": recall_at_k(
                retrieved_ids,
                relevance
            ),
            "ndcg_at_5": ndcg_at_k(
                retrieved_ids,
                relevance
            ),
            "unanswerable_non_relevant_rate": None
        }

    return {
        "precision_at_5": None,
        "recall_at_5": None,
        "ndcg_at_5": None,
        "unanswerable_non_relevant_rate": (
            unanswerable_non_relevant_rate(
                retrieved_ids,
                relevance
            )
        )
    }


# ==================================================
# 8. MAIN COMPARISON
# ==================================================

def evaluate():

    questions = load_json(QUESTIONS_FILE)
    labels = load_json(LABELS_FILE)

    strategies = {
        "Original RAG": [],
        "Multi-Query RAG": []
    }

    all_question_results = []

    print("\n" + "=" * 75)
    print("DOCUMIND AI - EXPANDED RETRIEVAL STRATEGY COMPARISON")
    print("=" * 75)

    print("Total questions:", len(questions))
    print("Candidate K:", CANDIDATE_K)
    print("Final Top K:", TOP_K)
    print("Search mode: Exact NumPy L2")

    for item in questions:

        question_id = item["id"]
        question = item["question"]
        expected_sources = item["expected_sources"]
        answerable = item["answerable"]

        relevance = labels.get(question_id, {})

        print("\n" + "=" * 75)
        print(question_id + ":", question)
        print("=" * 75)

        # Original strategy
        original_chunks = retrieve(
            question,
            [question],
            expected_sources
        )

        # Multi-query strategy
        queries = build_queries(question)

        multi_chunks = retrieve(
            question,
            queries,
            expected_sources
        )

        strategy_chunks = {
            "Original RAG": original_chunks,
            "Multi-Query RAG": multi_chunks
        }

        question_result = {
            "id": question_id,
            "question": question,
            "expected_sources": expected_sources,
            "answerable": answerable,
            "generated_queries": queries,
            "strategies": {}
        }

        for strategy_name, chunks in strategy_chunks.items():

            metrics = calculate_metrics(
                chunks,
                relevance,
                answerable
            )

            question_result["strategies"][strategy_name] = {
                "metrics": metrics,
                "retrieved_chunks": chunks
            }

            strategies[strategy_name].append(metrics)

            print("\n", strategy_name)

            print(
                "Retrieved IDs:",
                [chunk["chunk_id"] for chunk in chunks]
            )

            if answerable:

                print(
                    "Precision@5:",
                    round(metrics["precision_at_5"], 4)
                )

                print(
                    "Recall@5:",
                    round(metrics["recall_at_5"], 4)
                )

                print(
                    "NDCG@5:",
                    round(metrics["ndcg_at_5"], 4)
                )

            else:

                print(
                    "Unanswerable non-relevant rate:",
                    metrics["unanswerable_non_relevant_rate"]
                )

        all_question_results.append(question_result)

    # ==================================================
    # 9. FINAL AVERAGES
    # ==================================================

    summary = {}

    print("\n" + "=" * 75)
    print("FINAL STRATEGY COMPARISON")
    print("=" * 75)

    for strategy_name, results in strategies.items():

        answerable_results = [
            result
            for result in results
            if result["precision_at_5"] is not None
        ]

        unanswerable_results = [
            result
            for result in results
            if result["unanswerable_non_relevant_rate"] is not None
        ]

        if answerable_results:

            average_precision = sum(
                item["precision_at_5"]
                for item in answerable_results
            ) / len(answerable_results)

            average_recall = sum(
                item["recall_at_5"]
                for item in answerable_results
            ) / len(answerable_results)

            average_ndcg = sum(
                item["ndcg_at_5"]
                for item in answerable_results
            ) / len(answerable_results)

        else:

            average_precision = 0.0
            average_recall = 0.0
            average_ndcg = 0.0

        average_unanswerable_rate = None

        if unanswerable_results:

            average_unanswerable_rate = sum(
                item["unanswerable_non_relevant_rate"]
                for item in unanswerable_results
            ) / len(unanswerable_results)

        summary[strategy_name] = {
            "average_precision_at_5": average_precision,
            "average_recall_at_5": average_recall,
            "average_ndcg_at_5": average_ndcg,
            "average_unanswerable_non_relevant_rate": (
                average_unanswerable_rate
            )
        }

        print("\n", strategy_name)

        print(
            "Average Precision@5:",
            round(average_precision, 4)
        )

        print(
            "Average Recall@5:",
            round(average_recall, 4)
        )

        print(
            "Average NDCG@5:",
            round(average_ndcg, 4)
        )

        print(
            "Average unanswerable non-relevant rate:",
            (
                round(average_unanswerable_rate, 4)
                if average_unanswerable_rate is not None
                else "N/A"
            )
        )

    # ==================================================
    # 10. SAVE COMPARISON
    # ==================================================

    report = {
        "configuration": {
            "candidate_k": CANDIDATE_K,
            "top_k": TOP_K,
            "rrf_k": RRF_K,
            "search_mode": "exact",
            "reranking": "Cross-Encoder"
        },
        "summary": summary,
        "questions": all_question_results
    }

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            report,
            file,
            indent=4,
            ensure_ascii=False
        )

    print("\n" + "=" * 75)
    print("COMPARISON COMPLETED")
    print("=" * 75)
    print("Results saved to:", OUTPUT_FILE)


# ==================================================
# 11. EXECUTION
# ==================================================

if __name__ == "__main__":
    evaluate()
