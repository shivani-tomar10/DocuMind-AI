
import math

from backend.retrieval import (
    retrieve_with_rrf,
    load_reranker
)

from backend.llm import generate_search_queries


# ==================================================
# 1. CONFIGURATION
# ==================================================

EVALUATION_SOURCE = "sample.pdf"

TOP_K = 5
CANDIDATE_K = 40
RRF_K = 60

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
# 2. EVALUATION METRICS
# ==================================================

def precision_at_k(retrieved_ids, relevance, k=5):

    relevant_count = sum(
        1
        for chunk_id in retrieved_ids[:k]
        if relevance.get(chunk_id, 0) > 0
    )

    return relevant_count / k if k > 0 else 0.0


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


def ndcg_at_k(retrieved_ids, relevance, k=5):

    dcg = 0.0

    for rank, chunk_id in enumerate(
        retrieved_ids[:k],
        start=1
    ):

        score = relevance.get(chunk_id, 0)

        dcg += score / math.log2(rank + 1)

    ideal_scores = sorted(
        relevance.values(),
        reverse=True
    )[:k]

    idcg = sum(
        score / math.log2(rank + 1)
        for rank, score in enumerate(
            ideal_scores,
            start=1
        )
    )

    return dcg / idcg if idcg > 0 else 0.0


# ==================================================
# 3. GENERATE SEARCH QUERIES
# ==================================================

def build_search_queries(question):

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
# 4. STRATEGY 1: ORIGINAL RAG
# ==================================================

def original_rag(question):

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

    return results["ids"][0]


# ==================================================
# 5. STRATEGY 2: MULTI-QUERY RAG
# ==================================================

def multi_query_rag(queries):

    results = retrieve_with_rrf(
        queries=queries,
        top_k=TOP_K,
        candidate_k=CANDIDATE_K,
        rrf_k=RRF_K,
        per_document=False,
        rerank=True,
        source=EVALUATION_SOURCE,
        search_mode="exact"
    )

    return results["ids"][0]


# ==================================================
# 6. STRATEGY 3: MULTI-QUERY MAX-SCORE RERANKING
# ==================================================

def multi_query_max_score_rag(queries):

    # Retrieve the complete candidate set using RRF.
    candidates = retrieve_with_rrf(
        queries=queries,
        top_k=CANDIDATE_K,
        candidate_k=CANDIDATE_K,
        rrf_k=RRF_K,
        per_document=False,
        rerank=False,
        source=EVALUATION_SOURCE,
        search_mode="exact"
    )

    documents = candidates["documents"][0]

    if not documents:
        return []

    # Load the Cross-Encoder.
    reranker = load_reranker()

    # Calculate relevance scores for every query.
    query_scores = []

    for query in queries:

        pairs = [
            [query, document]
            for document in documents
        ]

        scores = reranker.predict(
            pairs,
            show_progress_bar=False
        )

        query_scores.append(scores)

    # Select the maximum score for each document.
    max_scores = [
        max(
            float(scores[index])
            for scores in query_scores
        )
        for index in range(len(documents))
    ]

    # Rank candidates using maximum relevance scores.
    ranked_indices = sorted(
        range(len(max_scores)),
        key=lambda index: max_scores[index],
        reverse=True
    )

    selected_indices = ranked_indices[:TOP_K]

    return [
        candidates["ids"][0][index]
        for index in selected_indices
    ]


# ==================================================
# 7. CALCULATE METRICS
# ==================================================

def calculate_metrics(retrieved_ids, relevance):

    return {
        "precision": precision_at_k(
            retrieved_ids,
            relevance,
            TOP_K
        ),

        "recall": recall_at_k(
            retrieved_ids,
            relevance,
            TOP_K
        ),

        "ndcg": ndcg_at_k(
            retrieved_ids,
            relevance,
            TOP_K
        )
    }


# ==================================================
# 8. MAIN EVALUATION
# ==================================================

def evaluate():

    strategy_results = {
        "Original RAG": [],
        "Multi-Query RAG": [],
        "Multi-Query Max-Score RAG": []
    }

    print("=" * 75)
    print("DOCUMIND AI - MULTI-QUERY RERANKING COMPARISON")
    print("=" * 75)

    print("Evaluation PDF:", EVALUATION_SOURCE)
    print("Top K:", TOP_K)
    print("Candidate K:", CANDIDATE_K)
    print("Evaluation questions:", len(RELEVANCE_LABELS))

    for question, relevance in RELEVANCE_LABELS.items():

        print("\n" + "=" * 75)
        print("QUESTION:", question)
        print("=" * 75)

        # Generate alternatives once so both multi-query
        # strategies use exactly the same queries.
        queries = build_search_queries(question)

        print("\nGenerated queries:")

        for query in queries:
            print("-", query)

        # Strategy 1
        original_ids = original_rag(question)

        # Strategy 2
        multi_ids = multi_query_rag(queries)

        # Strategy 3
        max_score_ids = multi_query_max_score_rag(
            queries
        )

        all_results = {
            "Original RAG": original_ids,
            "Multi-Query RAG": multi_ids,
            "Multi-Query Max-Score RAG": max_score_ids
        }

        for strategy, retrieved_ids in all_results.items():

            metrics = calculate_metrics(
                retrieved_ids,
                relevance
            )

            strategy_results[strategy].append(
                metrics
            )

            print("\n", strategy)
            print("Retrieved:", retrieved_ids)

            print(
                "Precision@5:",
                round(metrics["precision"], 4)
            )

            print(
                "Recall@5:",
                round(metrics["recall"], 4)
            )

            print(
                "NDCG@5:",
                round(metrics["ndcg"], 4)
            )

    # ==================================================
    # 9. FINAL COMPARISON
    # ==================================================

    print("\n" + "=" * 75)
    print("FINAL AVERAGE COMPARISON")
    print("=" * 75)

    for strategy, results in strategy_results.items():

        if not results:
            continue

        count = len(results)

        avg_precision = sum(
            item["precision"]
            for item in results
        ) / count

        avg_recall = sum(
            item["recall"]
            for item in results
        ) / count

        avg_ndcg = sum(
            item["ndcg"]
            for item in results
        ) / count

        print("\n", strategy)

        print(
            "Average Precision@5:",
            round(avg_precision, 4)
        )

        print(
            "Average Recall@5:",
            round(avg_recall, 4)
        )

        print(
            "Average NDCG@5:",
            round(avg_ndcg, 4)
        )

    print("\n" + "=" * 75)
    print("EVALUATION COMPLETED")
    print("=" * 75)


# ==================================================
# 10. RUN
# ==================================================

if __name__ == "__main__":
    evaluate()
