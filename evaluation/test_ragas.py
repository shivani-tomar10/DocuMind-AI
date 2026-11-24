
import os
import json
import asyncio

from dotenv import load_dotenv
from openai import OpenAI

from ragas.llms import llm_factory
from ragas.embeddings import HuggingFaceEmbeddings

from ragas.metrics.collections import (
    Faithfulness,
    ContextPrecision,
    ContextRecall,
    AnswerRelevancy
)

from backend.retrieval import retrieve_relevant_chunks
from backend.prompts import build_context
from backend.llm import generate_answer


# =====================================================
# 1. LOAD ENVIRONMENT VARIABLES
# =====================================================

load_dotenv(override=True)

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env")


# =====================================================
# 2. INITIALIZE GEMINI LLM
# =====================================================

client = OpenAI(
    api_key=api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

llm = llm_factory(
    model="gemini-3.5-flash-lite",
    provider="openai",
    client=client
)

print("Gemini LLM initialized successfully!")


# =====================================================
# 3. INITIALIZE HUGGING FACE EMBEDDINGS
# =====================================================

embeddings = HuggingFaceEmbeddings(
    model="all-MiniLM-L6-v2",
    normalize_embeddings=True
)

print("Hugging Face embeddings initialized successfully!")


# =====================================================
# 4. INITIALIZE RAGAS METRICS
# =====================================================

faithfulness = Faithfulness(llm=llm)

context_precision = ContextPrecision(llm=llm)

context_recall = ContextRecall(llm=llm)

answer_relevancy = AnswerRelevancy(
    llm=llm,
    embeddings=embeddings
)

print("All four Ragas metrics initialized successfully!")


# =====================================================
# 5. LOAD EVALUATION DATASET
# =====================================================

DATASET_PATH = "evaluation/test_dataset.json"

with open(DATASET_PATH, "r", encoding="utf-8") as file:
    dataset = json.load(file)

print("Dataset loaded successfully!")
print("Total questions:", len(dataset))


# =====================================================
# 6. EVALUATE ONE QUESTION
# =====================================================

async def evaluate_question(item):

    question_id = item["id"]
    question = item["question"]
    reference = item["reference_answer"]

    # Skip unanswerable questions
    if not item["answerable"] or not reference:
        print(f"\nSkipping {question_id}: No reference answer.")
        return None

    print(f"\n{'=' * 55}")
    print(f"Evaluating {question_id}: {question}")
    print("=" * 55)

    # Q8 needs retrieval from all uploaded documents
    search_all = question_id == "Q8"

    # Retrieve relevant chunks
    results = retrieve_relevant_chunks(
        question,
        top_k=3,
        per_document=search_all
    )

    contexts = results["documents"][0]

    if not contexts:
        raise ValueError("No relevant contexts retrieved.")

    # Construct context for the LLM
    context = build_context(results)

    # Generate RAG answer
    response = generate_answer(question, context)

    if not response or not response.strip():
        raise ValueError("LLM returned an empty answer.")

    print("\nGenerated Answer:")
    print(response)

    # -------------------------------------------------
    # 7. RUN FOUR RAGAS METRICS
    # -------------------------------------------------

    scores = {}

    # Faithfulness
    result = await faithfulness.ascore(
        user_input=question,
        response=response,
        retrieved_contexts=contexts
    )

    scores["faithfulness"] = float(result.value)

    # Context Precision
    result = await context_precision.ascore(
        user_input=question,
        reference=reference,
        retrieved_contexts=contexts
    )

    scores["context_precision"] = float(result.value)

    # Context Recall
    result = await context_recall.ascore(
        user_input=question,
        retrieved_contexts=contexts,
        reference=reference
    )

    scores["context_recall"] = float(result.value)

    # Answer Relevancy
    result = await answer_relevancy.ascore(
        user_input=question,
        response=response
    )

    scores["answer_relevancy"] = float(result.value)

    # -------------------------------------------------
    # 8. DISPLAY SCORES
    # -------------------------------------------------

    print("\nEvaluation Scores:")

    for metric, score in scores.items():
        print(f"{metric}: {score:.4f}")

    # -------------------------------------------------
    # 9. PREPARE RESULT
    # -------------------------------------------------

    evaluation_result = {
        "id": question_id,
        "question": question,
        "answer": response,
        "scores": scores
    }

    return evaluation_result


# =====================================================
# 10. EVALUATE COMPLETE DATASET
# =====================================================

async def main():

    evaluation_results = []

    failed_questions = []

    skipped_questions = []

    for item in dataset:

        try:

            result = await evaluate_question(item)

            if result is not None:
                evaluation_results.append(result)

            else:
                skipped_questions.append(item["id"])

        except Exception as error:

            print(f"\nEvaluation failed for {item['id']}: {error}")

            failed_questions.append(item["id"])

            print("\nWaiting 65 seconds to avoid API rate limits...")

            await asyncio.sleep(65)

    # =================================================
    # 11. SAVE RESULTS
    # =================================================

    RESULTS_PATH = "evaluation/ragas_results.json"

    with open(
        RESULTS_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            evaluation_results,
            file,
            indent=4,
            ensure_ascii=False
        )

    # =================================================
    # 12. FINAL EVALUATION SUMMARY
    # =================================================

    print("\n")
    print("=" * 55)
    print("             EVALUATION COMPLETE")
    print("=" * 55)

    print("Total dataset questions:", len(dataset))

    print("Successfully evaluated:", len(evaluation_results))

    print("Failed questions:", failed_questions)

    print("Skipped questions:", skipped_questions)

    # =================================================
    # 13. CALCULATE AVERAGE METRIC SCORES
    # =================================================

    if evaluation_results:

        metric_names = [
            "faithfulness",
            "context_precision",
            "context_recall",
            "answer_relevancy"
        ]

        print("\nAverage Ragas Scores:")
        print("-" * 40)

        for metric in metric_names:

            average = sum(
                result["scores"][metric]
                for result in evaluation_results
            ) / len(evaluation_results)

            print(f"{metric}: {average:.4f}")

    else:

        print("\nNo successful evaluations to calculate averages.")

    print("\nResults saved to:", RESULTS_PATH)


# =====================================================
# 14. RUN APPLICATION
# =====================================================

if __name__ == "__main__":

    asyncio.run(main())


DocuMind AI — Baseline Evaluation

Dataset: 10 questions
Evaluated: 7
Skipped: 3
Failed: 0

Faithfulness:       1.0000
Context Precision:  0.5357
Context Recall:     0.6429
Answer Relevancy:   0.8382