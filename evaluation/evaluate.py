
import json
from pathlib import Path

from backend.retrieval import retrieve_relevant_chunks


# ==================================================
# 1. LOAD EVALUATION DATASET
# ==================================================

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "test_dataset.json"

if not DATASET_PATH.exists():
    raise FileNotFoundError(
        f"Evaluation dataset not found: {DATASET_PATH}"
    )

with open(DATASET_PATH, "r", encoding="utf-8") as file:
    test_cases = json.load(file)

if not isinstance(test_cases, list):
    raise ValueError("Evaluation dataset must contain a list.")


# ==================================================
# 2. EVALUATION CONFIGURATION
# ==================================================

TOP_K = 3

SEARCH_MODE = "exact"

total_tests = 0
passed_tests = 0

total_relevant_chunks = 0
total_retrieved_chunks = 0

total_expected_sources = 0
total_found_sources = 0


# ==================================================
# 3. SOURCE MATCHING
# ==================================================

def source_matches(expected, actual):
    """
    Match expected source names against retrieved filenames.
    """

    expected = Path(expected).stem.lower().strip()
    actual = Path(actual).stem.lower().strip()

    return (
        expected == actual
        or expected in actual
        or actual in expected
    )


# ==================================================
# 4. RUN RETRIEVAL EVALUATION
# ==================================================

print("\n========== DOCUMIND AI RAG EVALUATION ==========\n")

print("Dataset:", DATASET_PATH.name)
print("Top K:", TOP_K)
print("Search mode:", SEARCH_MODE)


for test in test_cases:

    question_id = test["id"]
    question = test["question"]
    expected_sources = test["expected_sources"]
    answerable = test["answerable"]

    # Skip unsupported questions.
    if not answerable:
        print(
            f"\n{question_id}: SKIPPED "
            "(Unsupported question)"
        )
        continue

    total_tests += 1

    # Multi-document retrieval.
    per_document = len(expected_sources) > 1

    # Retrieve relevant chunks.
    results = retrieve_relevant_chunks(
        question,
        top_k=TOP_K,
        per_document=per_document,
        search_mode=SEARCH_MODE
    )

    retrieved_metadata = results["metadatas"][0]
    retrieved_documents = results["documents"][0]

    # Identify unique source documents.
    found_sources = {
        metadata["source"]
        for metadata in retrieved_metadata
        if metadata and metadata.get("source")
    }

    # Match expected sources.
    matched_sources = set()

    for expected in expected_sources:

        if any(
            source_matches(expected, actual)
            for actual in found_sources
        ):
            matched_sources.add(expected)

    # Source hit rate.
    source_hit = len(matched_sources) > 0

    # Source recall.
    source_recall = (
        len(matched_sources) / len(expected_sources)
        if expected_sources else 0
    )

    # Chunk-level source precision.
    relevant_chunks = sum(
        1
        for metadata in retrieved_metadata
        if metadata
        and any(
            source_matches(expected, metadata.get("source", ""))
            for expected in expected_sources
        )
    )

    chunk_precision = (
        relevant_chunks / len(retrieved_metadata)
        if retrieved_metadata else 0
    )

    # Update aggregate metrics.
    total_relevant_chunks += relevant_chunks
    total_retrieved_chunks += len(retrieved_metadata)

    total_expected_sources += len(expected_sources)
    total_found_sources += len(matched_sources)

    if source_hit:
        passed_tests += 1

    # Display results.
    print("\n" + "-" * 60)
    print(f"Question ID: {question_id}")
    print("Question:", question)

    print("Expected Sources:", expected_sources)
    print("Retrieved Sources:", sorted(found_sources))

    print("Retrieved Chunks:", len(retrieved_documents))
    print("Relevant Chunks:", relevant_chunks)

    print(f"Source Recall: {source_recall:.2%}")
    print(f"Source Precision: {chunk_precision:.2%}")

    print("Result:", "PASS" if source_hit else "FAIL")


# ==================================================
# 5. FINAL EVALUATION METRICS
# ==================================================

print("\n========== FINAL EVALUATION ==========\n")

if total_tests > 0:

    hit_rate = passed_tests / total_tests

    source_recall = (
        total_found_sources / total_expected_sources
        if total_expected_sources else 0
    )

    chunk_precision = (
        total_relevant_chunks / total_retrieved_chunks
        if total_retrieved_chunks else 0
    )

    print("Answerable Tests:", total_tests)
    print(f"Passed Tests: {passed_tests}/{total_tests}")

    print(f"Source Hit Rate: {hit_rate:.2%}")
    print(f"Source Recall: {source_recall:.2%}")
    print(f"Source Precision: {chunk_precision:.2%}")

else:
    print("No answerable test cases found.")

print("\nEvaluation completed!")
