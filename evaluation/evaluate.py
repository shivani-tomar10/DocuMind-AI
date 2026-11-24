
import json
from pathlib import Path

from backend.retrieval import retrieve_relevant_chunks


# --------------------------------
# Load Evaluation Dataset
# --------------------------------

BASE_DIR = Path(__file__).resolve().parent

DATASET_PATH = BASE_DIR / "test_dataset.json"


with open(DATASET_PATH, "r", encoding="utf-8") as file:
    test_cases = json.load(file)


# --------------------------------
# Evaluation Configuration
# --------------------------------

TOP_K = 3

total_tests = 0
passed_tests = 0

total_relevant_chunks = 0
total_retrieved_chunks = 0
total_expected_sources = 0
total_found_sources = 0


# --------------------------------
# Run Retrieval Evaluation
# --------------------------------

print("\n========== DOCUMIND AI RAG EVALUATION ==========\n")


for test in test_cases:

    question_id = test["id"]
    question = test["question"]
    expected_sources = test["expected_sources"]
    answerable = test["answerable"]

    # Skip unsupported questions in source-retrieval scoring
    if not answerable:
        print(f"{question_id}: SKIPPED (Unsupported question)")
        continue

    total_tests += 1

    # Multi-document test uses document-wise retrieval
    per_document = len(expected_sources) > 1

    # Retrieve relevant chunks
    results = retrieve_relevant_chunks(
        question,
        top_k=TOP_K,
        per_document=per_document
    )

    retrieved_sources = results["metadatas"][0]
    retrieved_documents = results["documents"][0]

    # Identify unique source documents
    found_sources = {
        metadata["source"]
        for metadata in retrieved_sources
        if metadata and metadata.get("source")
    }

    # Match expected source names
    matched_sources = set()

    for expected in expected_sources:

        for actual in found_sources:

            if expected.lower() in actual.lower():
                matched_sources.add(expected)

    # Calculate source coverage
    source_hit = len(matched_sources) > 0

    source_recall = (
        len(matched_sources) / len(expected_sources)
        if expected_sources else 0
    )

    # Calculate chunk-level source precision
    relevant_chunks = sum(
        1
        for metadata in retrieved_sources
        if metadata
        and any(
            expected.lower() in metadata.get("source", "").lower()
            for expected in expected_sources
        )
    )

    chunk_precision = (
        relevant_chunks / len(retrieved_sources)
        if retrieved_sources else 0
    )

    # Update aggregate metrics
    total_relevant_chunks += relevant_chunks
    total_retrieved_chunks += len(retrieved_sources)

    total_expected_sources += len(expected_sources)
    total_found_sources += len(matched_sources)

    if source_hit:
        passed_tests += 1

    # Display individual test results
    print(f"\n{question_id}: {question}")

    print("Expected Sources:", expected_sources)
    print("Retrieved Sources:", list(found_sources))

    print("Relevant Chunks:", relevant_chunks)
    print("Total Retrieved Chunks:", len(retrieved_documents))

    print(f"Source Recall: {source_recall:.2%}")
    print(f"Source Precision: {chunk_precision:.2%}")

    print("Result:", "PASS" if source_hit else "FAIL")


# --------------------------------
# Final Evaluation Metrics
# --------------------------------

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

    print(f"Answerable Tests: {total_tests}")
    print(f"Source Hit Rate: {hit_rate:.2%}")
    print(f"Source Recall: {source_recall:.2%}")
    print(f"Source Precision: {chunk_precision:.2%}")

else:

    print("No answerable test cases found.")

print("\nEvaluation completed!")
