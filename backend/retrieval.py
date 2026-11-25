
import logging
from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder

from backend.vector_store import collection

logger = logging.getLogger(__name__)


# ==================================================
# 1. LAZY LOAD MODELS
# ==================================================

@lru_cache(maxsize=1)
def load_embedding_model():
    logger.info("Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    logger.info("Embedding model loaded successfully!")
    return model


@lru_cache(maxsize=1)
def load_reranker():
    logger.info("Loading cross-encoder reranker...")
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    logger.info("Cross-encoder loaded successfully!")
    return reranker


# ==================================================
# 2. EMPTY RESULTS STRUCTURE
# ==================================================

def empty_results():
    return {
        "ids": [[]],
        "embeddings": None,
        "documents": [[]],
        "metadatas": [[]],
        "distances": [[]],
        "rerank_scores": [[]]
    }


# ==================================================
# 3. EXACT NUMPY RETRIEVAL
# ==================================================

def retrieve_exact(
    query_embedding,
    top_k=5,
    source=None
):
    """
    Retrieve nearest chunks using exact L2 distance.

    This bypasses ChromaDB's approximate nearest-neighbor
    search while preserving the existing collection.
    """

    if top_k <= 0:
        return empty_results()

    where_filter = {"source": source} if source else None

    stored = collection.get(
        where=where_filter,
        include=[
            "embeddings",
            "documents",
            "metadatas"
        ]
    )

    if not stored["ids"]:
        return empty_results()

    vectors = np.asarray(
        stored["embeddings"],
        dtype=np.float32
    )

    query_vector = np.asarray(
        query_embedding,
        dtype=np.float32
    ).reshape(-1)

    if vectors.ndim != 2 or vectors.shape[1] != len(query_vector):
        raise ValueError(
            "Query and stored embedding dimensions do not match."
        )

    # Calculate exact Euclidean distances.
    distances = np.linalg.norm(
        vectors - query_vector,
        axis=1
    )

    # Select the nearest chunks.
    selected_indices = np.argsort(distances)[:top_k]

    return {
        "ids": [[stored["ids"][i] for i in selected_indices]],
        "embeddings": None,
        "documents": [[stored["documents"][i] for i in selected_indices]],
        "metadatas": [[stored["metadatas"][i] for i in selected_indices]],
        "distances": [[float(distances[i]) for i in selected_indices]],
        "rerank_scores": [[]]
    }


# ==================================================
# 4. SEMANTIC RETRIEVAL
# ==================================================

def retrieve_relevant_chunks(
    query,
    top_k=3,
    per_document=False,
    source=None,
    search_mode="hnsw"
):
    """
    Retrieve relevant chunks from ChromaDB.

    search_mode:
        hnsw  -> Existing ChromaDB vector search.
        exact -> Exact NumPy L2 search.

    source:
        Optional PDF source filter.

    per_document:
        Retrieve top_k chunks separately per PDF.
    """

    if search_mode not in {"hnsw", "exact"}:
        raise ValueError(
            "search_mode must be 'hnsw' or 'exact'."
        )

    if top_k <= 0:
        return empty_results()

    total_documents = collection.count()

    if total_documents == 0:
        return empty_results()

    model = load_embedding_model()

    query_embedding = model.encode(
        [query]
    ).tolist()

    # Exact NumPy search.
    if search_mode == "exact":

        if not per_document:
            return retrieve_exact(
                query_embedding[0],
                top_k=top_k,
                source=source
            )

        all_results = collection.get(
            where={"source": source} if source else None,
            include=["metadatas"]
        )

        sources = sorted({
            metadata["source"]
            for metadata in all_results["metadatas"]
        })

        combined = empty_results()

        for current_source in sources:

            results = retrieve_exact(
                query_embedding[0],
                top_k=top_k,
                source=current_source
            )

            for key in [
                "ids",
                "documents",
                "metadatas",
                "distances"
            ]:
                combined[key][0].extend(results[key][0])

        return combined

    # Existing HNSW retrieval.
    where_filter = {"source": source} if source else None

    if not per_document:

        arguments = {
            "query_embeddings": query_embedding,
            "n_results": min(top_k, total_documents),
            "include": [
                "documents",
                "metadatas",
                "distances"
            ]
        }

        if where_filter:
            arguments["where"] = where_filter

        return collection.query(**arguments)

    # Per-document HNSW retrieval.
    all_sources = collection.get(
        where=where_filter,
        include=["metadatas"]
    )

    sources = sorted({
        metadata["source"]
        for metadata in all_sources["metadatas"]
    })

    combined = empty_results()

    for current_source in sources:

        source_count = sum(
            1 for metadata in all_sources["metadatas"]
            if metadata["source"] == current_source
        )

        results = collection.query(
            query_embeddings=query_embedding,
            n_results=min(top_k, source_count),
            where={"source": current_source},
            include=[
                "documents",
                "metadatas",
                "distances"
            ]
        )

        for key in [
            "ids",
            "documents",
            "metadatas",
            "distances"
        ]:
            combined[key][0].extend(results[key][0])

    ranked = sorted(
        zip(
            combined["ids"][0],
            combined["documents"][0],
            combined["metadatas"][0],
            combined["distances"][0]
        ),
        key=lambda item: item[3]
    )

    return {
        "ids": [[item[0] for item in ranked]],
        "embeddings": None,
        "documents": [[item[1] for item in ranked]],
        "metadatas": [[item[2] for item in ranked]],
        "distances": [[item[3] for item in ranked]],
        "rerank_scores": [[]]
    }


# ==================================================
# 5. CROSS-ENCODER RERANKING
# ==================================================

def rerank_chunks(
    query,
    results,
    top_k=10,
    per_document=False
):

    documents = results["documents"][0]

    if not documents:
        return empty_results()

    reranker = load_reranker()

    pairs = [
        [query, document]
        for document in documents
    ]

    scores = reranker.predict(
        pairs,
        show_progress_bar=False
    )

    ranked_indices = sorted(
        range(len(scores)),
        key=lambda i: float(scores[i]),
        reverse=True
    )

    if per_document:

        source_counts = {}
        selected_indices = []

        for index in ranked_indices:

            current_source = results["metadatas"][0][index]["source"]
            count = source_counts.get(current_source, 0)

            if count < top_k:
                selected_indices.append(index)
                source_counts[current_source] = count + 1

    else:
        selected_indices = ranked_indices[:top_k]

    output = empty_results()

    for index in selected_indices:

        for key in [
            "ids",
            "documents",
            "metadatas",
            "distances"
        ]:
            output[key][0].append(
                results[key][0][index]
            )

        output["rerank_scores"][0].append(
            float(scores[index])
        )

    return output


# ==================================================
# 6. MULTI-QUERY RETRIEVAL USING RRF
# ==================================================

def retrieve_with_rrf(
    queries,
    top_k=10,
    candidate_k=40,
    rrf_k=60,
    per_document=False,
    rerank=True,
    source=None,
    search_mode="hnsw"
):
    """
    Combine multiple retrieval queries using Reciprocal Rank Fusion.

    Duplicate normalized text is removed before reranking.

    When per_document=True, identical text from different PDFs
    is preserved as a separate result for each PDF.
    """

    if not queries:
        return empty_results()

    if top_k <= 0 or candidate_k <= 0:
        return empty_results()

    scores = {}
    chunk_data = {}

    # Step 1: Retrieve candidates for every query.
    for query in queries:

        if not query or not query.strip():
            continue

        results = retrieve_relevant_chunks(
            query,
            top_k=candidate_k,
            per_document=per_document,
            source=source,
            search_mode=search_mode
        )

        ids = results["ids"][0]
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        # Step 2: Calculate Reciprocal Rank Fusion scores.
        for rank, (
            chunk_id,
            document,
            metadata,
            distance
        ) in enumerate(
            zip(ids, documents, metadatas, distances),
            start=1
        ):

            scores[chunk_id] = (
                scores.get(chunk_id, 0)
                + 1 / (rrf_k + rank)
            )

            if chunk_id not in chunk_data:
                chunk_data[chunk_id] = {
                    "document": document,
                    "metadata": metadata,
                    "distance": distance
                }

    # Step 3: Handle empty retrieval results.
    if not scores:
        return empty_results()

    # Step 4: Sort candidates by their RRF scores.
    ranked_ids = sorted(
        scores,
        key=lambda chunk_id: scores[chunk_id],
        reverse=True
    )

    # Step 5: Preserve candidates from each document.
    if per_document:

        source_groups = {}

        for chunk_id in ranked_ids:

            current_source = chunk_data[chunk_id]["metadata"]["source"]

            source_groups.setdefault(
                current_source,
                []
            ).append(chunk_id)

        ranked_ids = [
            chunk_id
            for group in source_groups.values()
            for chunk_id in group[:candidate_k]
        ]

    else:
        ranked_ids = ranked_ids[:candidate_k]

    # Step 6: Remove duplicate text.
    unique_ids = []
    seen_documents = set()

    for chunk_id in ranked_ids:

        document = chunk_data[chunk_id]["document"]
        metadata = chunk_data[chunk_id]["metadata"]

        normalized_document = " ".join(
            document.split()
        ).casefold()

        # Keep identical text separate across different PDFs
        # when per-document retrieval is enabled.
        if per_document:
            deduplication_key = (
                metadata["source"],
                normalized_document
            )
        else:
            deduplication_key = normalized_document

        if deduplication_key in seen_documents:
            continue

        seen_documents.add(deduplication_key)
        unique_ids.append(chunk_id)

    # Step 7: Build fused retrieval results.
    fused = empty_results()

    for chunk_id in unique_ids:

        data = chunk_data[chunk_id]

        fused["ids"][0].append(chunk_id)
        fused["documents"][0].append(data["document"])
        fused["metadatas"][0].append(data["metadata"])
        fused["distances"][0].append(data["distance"])

    if not fused["ids"][0]:
        return empty_results()

    # Step 8: Cross-Encoder reranking.
    if rerank:

        return rerank_chunks(
            query=queries[0],
            results=fused,
            top_k=top_k,
            per_document=per_document
        )

    # Step 9: Select final results without reranking.
    if per_document:

        source_counts = {}
        selected_indices = []

        for index, metadata in enumerate(fused["metadatas"][0]):

            current_source = metadata["source"]
            count = source_counts.get(current_source, 0)

            if count < top_k:
                selected_indices.append(index)
                source_counts[current_source] = count + 1

    else:
        selected_indices = list(
            range(min(top_k, len(fused["ids"][0])))
        )

    # Step 10: Build final output.
    output = empty_results()

    for index in selected_indices:

        for key in [
            "ids",
            "documents",
            "metadatas",
            "distances"
        ]:
            output[key][0].append(
                fused[key][0][index]
            )

    return output


# ==================================================
# 7. SAME-PAGE CHUNK EXPANSION
# ==================================================

def expand_retrieved_chunks(results):

    expanded = empty_results()
    seen_ids = set()

    original_distances = dict(
        zip(results["ids"][0], results["distances"][0])
    )

    original_scores = dict(
        zip(
            results["ids"][0],
            results.get("rerank_scores", [[]])[0]
        )
    )

    for i, chunk_id in enumerate(results["ids"][0]):

        metadata = results["metadatas"][0][i]

        page_chunks = collection.get(
            where={
                "$and": [
                    {"source": metadata["source"]},
                    {"page_number": metadata["page_number"]}
                ]
            },
            include=["documents", "metadatas"]
        )

        sorted_chunks = sorted(
            zip(
                page_chunks["ids"],
                page_chunks["documents"],
                page_chunks["metadatas"]
            ),
            key=lambda item: int(item[0].split("_chunk")[-1])
        )

        for neighbor_id, document, neighbor_metadata in sorted_chunks:

            if neighbor_id in seen_ids:
                continue

            expanded["ids"][0].append(neighbor_id)
            expanded["documents"][0].append(document)
            expanded["metadatas"][0].append(neighbor_metadata)

            expanded["distances"][0].append(
                original_distances.get(neighbor_id)
            )

            expanded["rerank_scores"][0].append(
                original_scores.get(neighbor_id)
            )

            seen_ids.add(neighbor_id)

    return expanded


# ==================================================
# 8. TESTING
# ==================================================

if __name__ == "__main__":

    query = "What is the Turing Test?"

    results = retrieve_relevant_chunks(
        query,
        top_k=5,
        source="sample.pdf",
        search_mode="exact"
    )

    print("Exact search results:")

    for chunk_id, metadata, distance in zip(
        results["ids"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        print(
            chunk_id,
            metadata["source"],
            metadata["page_number"],
            round(distance, 4)
        )
