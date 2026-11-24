from functools import lru_cache

from sentence_transformers import SentenceTransformer, CrossEncoder

from backend.vector_store import collection


# ==================================================
# 1. LAZY LOAD MODELS
# ==================================================

@lru_cache(maxsize=1)
def load_embedding_model():
    """Load and cache the embedding model when first needed."""
    print("Loading embedding model...", flush=True)

    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("Embedding model loaded successfully!", flush=True)

    return model


@lru_cache(maxsize=1)
def load_reranker():
    """Load and cache the cross-encoder when first needed."""
    print("Loading cross-encoder reranker...", flush=True)

    reranker = CrossEncoder(
        "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )

    print("Cross-encoder loaded successfully!", flush=True)

    return reranker


# ==================================================
# 2. EMPTY RESULTS STRUCTURE
# ==================================================

def empty_results():
    return {
        "ids": [[]],
        "documents": [[]],
        "metadatas": [[]],
        "distances": [[]],
        "rerank_scores": [[]]
    }


# ==================================================
# 3. SEMANTIC RETRIEVAL
# ==================================================

def retrieve_relevant_chunks(
    query,
    top_k=3,
    per_document=False
):
    """
    Retrieve relevant chunks from ChromaDB.

    per_document=False:
        Search globally across all PDFs.

    per_document=True:
        Retrieve top_k chunks separately
        from every uploaded PDF.
    """

    total_documents = collection.count()

    if total_documents == 0:
        print("No documents available in the database.")
        return empty_results()

    # Load embedding model only when required
    model = load_embedding_model()

    query_embedding = model.encode(
        [query]
    ).tolist()

    # Global semantic search
    if not per_document:
        return collection.query(
            query_embeddings=query_embedding,
            n_results=min(top_k, total_documents),
            include=[
                "documents",
                "metadatas",
                "distances"
            ]
        )

    # Per-document semantic search
    all_sources = collection.get(
        include=["metadatas"]
    )

    sources = sorted({
        metadata["source"]
        for metadata in all_sources["metadatas"]
    })

    combined_results = empty_results()

    for source in sources:
        source_count = sum(
            1 for metadata in all_sources["metadatas"]
            if metadata["source"] == source
        )

        results = collection.query(
            query_embeddings=query_embedding,
            n_results=min(top_k, source_count),
            where={"source": source},
            include=[
                "documents",
                "metadatas",
                "distances"
            ]
        )

        combined_results["ids"][0].extend(
            results["ids"][0]
        )

        combined_results["documents"][0].extend(
            results["documents"][0]
        )

        combined_results["metadatas"][0].extend(
            results["metadatas"][0]
        )

        combined_results["distances"][0].extend(
            results["distances"][0]
        )

    ranked_results = sorted(
        zip(
            combined_results["ids"][0],
            combined_results["documents"][0],
            combined_results["metadatas"][0],
            combined_results["distances"][0]
        ),
        key=lambda item: item[3]
    )

    return {
        "ids": [[item[0] for item in ranked_results]],
        "documents": [[item[1] for item in ranked_results]],
        "metadatas": [[item[2] for item in ranked_results]],
        "distances": [[item[3] for item in ranked_results]],
        "rerank_scores": [[]]
    }


# ==================================================
# 4. CROSS-ENCODER RERANKING
# ==================================================

def rerank_chunks(
    query,
    results,
    top_k=10,
    per_document=False
):
    """
    Rerank retrieved chunks using a cross-encoder.

    Higher scores indicate stronger relevance.

    per_document=True:
        Preserve up to top_k chunks per PDF.
    """

    documents = results["documents"][0]

    if not documents:
        return empty_results()

    # Load reranker only when reranking is requested
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
        key=lambda index: float(scores[index]),
        reverse=True
    )

    # Source-balanced selection
    if per_document:
        source_counts = {}
        selected_indices = []

        for index in ranked_indices:
            source = results["metadatas"][0][index]["source"]

            count = source_counts.get(source, 0)

            if count < top_k:
                selected_indices.append(index)
                source_counts[source] = count + 1

    else:
        selected_indices = ranked_indices[:top_k]

    reranked_results = empty_results()

    for index in selected_indices:
        reranked_results["ids"][0].append(
            results["ids"][0][index]
        )

        reranked_results["documents"][0].append(
            documents[index]
        )

        reranked_results["metadatas"][0].append(
            results["metadatas"][0][index]
        )

        reranked_results["distances"][0].append(
            results["distances"][0][index]
        )

        reranked_results["rerank_scores"][0].append(
            float(scores[index])
        )

    return reranked_results


# ==================================================
# 5. MULTI-QUERY RETRIEVAL USING RRF
# ==================================================

def retrieve_with_rrf(
    queries,
    top_k=10,
    candidate_k=40,
    rrf_k=60,
    per_document=False,
    rerank=True
):
    """
    Retrieve chunks using multiple queries
    and combine their rankings using RRF.

    candidate_k:
        Number of candidates retained after RRF
        before reranking.

    top_k:
        Final number of chunks returned.

    rerank=True:
        Apply cross-encoder to the candidate pool.

    rerank=False:
        Return the original RRF top_k results.
    """

    scores = {}
    chunk_data = {}

    # Execute semantic retrieval for each query
    for query in queries:
        results = retrieve_relevant_chunks(
            query,
            top_k=candidate_k,
            per_document=per_document
        )

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

            # Reciprocal Rank Fusion
            rrf_score = 1 / (rrf_k + rank)

            scores[chunk_id] = (
                scores.get(chunk_id, 0) + rrf_score
            )

            if chunk_id not in chunk_data:
                chunk_data[chunk_id] = {
                    "document": document,
                    "metadata": metadata,
                    "distance": distance
                }

    if not scores:
        return empty_results()

    # Sort by combined RRF score
    ranked_ids = sorted(
        scores,
        key=lambda chunk_id: scores[chunk_id],
        reverse=True
    )

    # Preserve candidate pool
    if per_document:
        source_groups = {}

        for chunk_id in ranked_ids:
            source = chunk_data[chunk_id]["metadata"]["source"]

            if source not in source_groups:
                source_groups[source] = []

            source_groups[source].append(chunk_id)

        candidate_ids = []

        for source_chunk_ids in source_groups.values():
            candidate_ids.extend(
                source_chunk_ids[:candidate_k]
            )

        candidate_set = set(candidate_ids)

        ranked_ids = [
            chunk_id
            for chunk_id in ranked_ids
            if chunk_id in candidate_set
        ]

    else:
        ranked_ids = ranked_ids[:candidate_k]

    # Construct candidate results
    fused_results = empty_results()

    for chunk_id in ranked_ids:
        data = chunk_data[chunk_id]

        fused_results["ids"][0].append(chunk_id)

        fused_results["documents"][0].append(
            data["document"]
        )

        fused_results["metadatas"][0].append(
            data["metadata"]
        )

        fused_results["distances"][0].append(
            data["distance"]
        )

    # Apply cross-encoder reranking
    if rerank:
        if not queries:
            return empty_results()

        return rerank_chunks(
            query=queries[0],
            results=fused_results,
            top_k=top_k,
            per_document=per_document
        )

    # Original RRF top_k selection
    if per_document:
        source_counts = {}
        selected_indices = []

        for index, metadata in enumerate(
            fused_results["metadatas"][0]
        ):
            source = metadata["source"]
            count = source_counts.get(source, 0)

            if count < top_k:
                selected_indices.append(index)
                source_counts[source] = count + 1

    else:
        selected_indices = list(
            range(min(top_k, len(fused_results["ids"][0])))
        )

    original_results = empty_results()

    for index in selected_indices:
        for key in [
            "ids",
            "documents",
            "metadatas",
            "distances"
        ]:
            original_results[key][0].append(
                fused_results[key][0][index]
            )

    return original_results


# ==================================================
# 6. SAME-PAGE CHUNK EXPANSION
# ==================================================

def expand_retrieved_chunks(results):
    """
    Expand retrieved chunks using all chunks
    from the same PDF page.
    """

    expanded_results = empty_results()

    seen_ids = set()

    original_distances = {
        chunk_id: distance
        for chunk_id, distance in zip(
            results["ids"][0],
            results["distances"][0]
        )
    }

    original_scores = {
        chunk_id: score
        for chunk_id, score in zip(
            results["ids"][0],
            results.get("rerank_scores", [[]])[0]
        )
    }

    for i, chunk_id in enumerate(
        results["ids"][0]
    ):
        metadata = results["metadatas"][0][i]

        source = metadata["source"]
        page_number = metadata["page_number"]

        page_chunks = collection.get(
            where={
                "$and": [
                    {"source": source},
                    {"page_number": page_number}
                ]
            },
            include=[
                "documents",
                "metadatas"
            ]
        )

        sorted_chunks = sorted(
            zip(
                page_chunks["ids"],
                page_chunks["documents"],
                page_chunks["metadatas"]
            ),
            key=lambda item: int(
                item[0].split("_chunk")[-1]
            )
        )

        for (
            neighbor_id,
            document,
            neighbor_metadata
        ) in sorted_chunks:

            if neighbor_id not in seen_ids:
                expanded_results["ids"][0].append(
                    neighbor_id
                )

                expanded_results["documents"][0].append(
                    document
                )

                expanded_results["metadatas"][0].append(
                    neighbor_metadata
                )

                expanded_results["distances"][0].append(
                    original_distances.get(neighbor_id)
                )

                expanded_results["rerank_scores"][0].append(
                    original_scores.get(neighbor_id)
                )

                seen_ids.add(neighbor_id)

    return expanded_results


# ==================================================
# 7. TESTING: ORIGINAL RRF VS RERANKED RRF
# ==================================================

if __name__ == "__main__":

    query = "What is Artificial Intelligence?"

    queries = [
        query,
        (
            "Definition and explanation of Artificial Intelligence, "
            "its purpose, goals and fundamental characteristics."
        )
    ]

    FINAL_TOP_K = 20
    CANDIDATE_K = 40

    # Test 1: Original RRF top 20
    print("\nORIGINAL RRF RANKING")

    original_results = retrieve_with_rrf(
        queries=queries,
        top_k=FINAL_TOP_K,
        candidate_k=CANDIDATE_K,
        per_document=False,
        rerank=False
    )

    print(
        "\nOriginal RRF result count:",
        len(original_results["ids"][0])
    )

    for index, (
        document,
        metadata,
        distance
    ) in enumerate(
        zip(
            original_results["documents"][0],
            original_results["metadatas"][0],
            original_results["distances"][0]
        ),
        start=1
    ):

        print(f"\nRank: {index}")
        print("Source:", metadata["source"])
        print("Page:", metadata["page_number"])
        print("Distance:", round(distance, 4))
        print("Content:", document[:250])

    # Test 2: RRF + Cross-Encoder top 20
    print("\n\nCROSS-ENCODER RERANKED RESULTS")

    reranked_results = retrieve_with_rrf(
        queries=queries,
        top_k=FINAL_TOP_K,
        candidate_k=CANDIDATE_K,
        per_document=False,
        rerank=True
    )

    print(
        "\nReranked result count:",
        len(reranked_results["ids"][0])
    )

    for index, (
        document,
        metadata,
        distance,
        score
    ) in enumerate(
        zip(
            reranked_results["documents"][0],
            reranked_results["metadatas"][0],
            reranked_results["distances"][0],
            reranked_results["rerank_scores"][0]
        ),
        start=1
    ):

        print(f"\nRank: {index}")
        print("Source:", metadata["source"])
        print("Page:", metadata["page_number"])
        print("L2 Distance:", round(distance, 4))
        print("Rerank Score:", round(score, 4))
        print("Content:", document[:250])

    # Test 3: Context expansion
    print("\n\nEXPANDING RERANKED CHUNKS")

    expanded_results = expand_retrieved_chunks(
        reranked_results
    )

    print(
        "\nTotal expanded chunks:",
        len(expanded_results["ids"][0])
    )

    source_counts = {}

    for metadata in expanded_results["metadatas"][0]:
        source = metadata["source"]

        source_counts[source] = (
            source_counts.get(source, 0) + 1
        )

    print("\nExpanded chunks per PDF:")

    for source, count in source_counts.items():
        print(f"{source}: {count}")
