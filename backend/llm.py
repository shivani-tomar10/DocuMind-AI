
import os
import re

from dotenv import load_dotenv
from google import genai


# --------------------------------
# 1. Load Gemini API Configuration
# --------------------------------

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env file")

client = genai.Client(api_key=api_key)


# --------------------------------
# 2. Generate Multiple Search Queries
# --------------------------------

def generate_search_queries(query, num_queries=3):
    """
    Generate multiple focused search queries
    for improved RAG retrieval.
    """

    if not query or not query.strip():
        return []

    if num_queries < 1:
        return []

    prompt = f"""
You are a search query optimization assistant
for a document-based RAG system.

Generate exactly {num_queries} different search queries
based on the user's original question.

STRICT RULES:

1. Preserve the original question's intent.
2. Keep the main subject and important keywords.
3. Use different natural wording and synonyms.
4. Explore different semantic formulations of the
   same question.
5. Do not introduce unrelated subtopics.
6. Do not invent facts or answers.
7. Keep every query concise.
8. Make the queries meaningfully different.
9. Return one query per line.
10. Do not include numbering, bullets, headings,
    explanations, or quotation marks.

USER QUESTION:
{query}

ALTERNATIVE SEARCH QUERIES:
"""

    try:
        interaction = client.interactions.create(
            model="gemini-3.5-flash-lite",
            input=prompt,
            timeout=60
        )

        output = interaction.output_text.strip()

        if not output:
            return []

        generated_queries = []

        for line in output.splitlines():

            cleaned = line.strip()

            # Remove numbering and bullet formatting
            cleaned = re.sub(
                r"^\s*(?:\d+[.)]\s*|[-*•]\s*)",
                "",
                cleaned
            )

            # Remove surrounding quotation marks
            cleaned = cleaned.strip("\"'` ")

            if not cleaned:
                continue

            # Avoid duplicate original query
            if cleaned.casefold() == query.strip().casefold():
                continue

            # Avoid duplicate generated queries
            if cleaned.casefold() not in {
                item.casefold()
                for item in generated_queries
            }:
                generated_queries.append(cleaned)

            if len(generated_queries) >= num_queries:
                break

        return generated_queries

    except Exception as error:

        print(
            "Multiple query generation failed:",
            type(error).__name__,
            str(error)
        )

        return []


# --------------------------------
# 3. Backward-Compatible Query Function
# --------------------------------

def generate_search_query(query):
    """
    Generate one alternative search query.
    Retained for backward compatibility.
    """

    queries = generate_search_queries(
        query,
        num_queries=1
    )

    if queries:
        return queries[0]

    return query


# --------------------------------
# 4. Generate Final RAG Answer
# --------------------------------

def generate_answer(query, context):
    """
    Generate an answer using retrieved document context.
    """

    if not context or not context.strip():
        return (
            "The information was not found "
            "in the uploaded documents."
        )

    prompt = f"""
You are DocuMind AI, an intelligent
multi-PDF research assistant.

Answer the user's question using ONLY
the information provided in the context.

STRICT RULES:

1. Answer the user's original question directly.
2. Do not use outside knowledge.
3. Do not invent facts or information.
4. If the answer is not available in the context,
   respond exactly:
   "The information was not found in the uploaded documents."
5. Provide clear and well-organized explanations.
6. Include relevant details from the context.
7. Cite the source and page number for factual claims.
8. Use this citation format:
   [Source: filename, Page: page_number]
9. Never invent source names or page numbers.
10. Only cite filenames and page numbers explicitly
    provided in the document context.
11. Avoid repeating the same information.

DOCUMENT CONTEXT:

{context}

USER QUESTION:

{query}

ANSWER:
"""

    try:
        interaction = client.interactions.create(
            model="gemini-3.5-flash-lite",
            input=prompt,
            timeout=60
        )

        answer = interaction.output_text.strip()

        if not answer:
            return (
                "Sorry, I could not generate "
                "an answer from the provided documents."
            )

        return answer

    except Exception as error:

        print(
            "Answer generation failed:",
            type(error).__name__,
            str(error)
        )

        return (
            "Sorry, I could not generate an answer at this time."
        )


# --------------------------------
# 5. Validate Generated Citations
# --------------------------------

def validate_citations(answer, metadatas):
    """
    Validate generated source filenames and page numbers
    against retrieved document metadata.

    Note:
    This validates citation references, not whether
    the cited passage semantically supports the claim.
    """

    valid_sources = {
        (
            str(metadata.get("source", "")).strip(),
            str(metadata.get("page_number", "")).strip()
        )
        for metadata in (metadatas or [])
        if metadata.get("source") is not None
        and metadata.get("page_number") is not None
    }

    citation_pattern = (
        r"\[Source:\s*(.*?),\s*Page:\s*(\d+)\]"
    )

    invalid_citations = []

    def check_citation(match):

        source = match.group(1).strip()
        page = match.group(2).strip()

        citation = (source, page)

        if citation in valid_sources:
            return match.group(0)

        invalid_citations.append(match.group(0))

        return "[Unverified citation]"

    validated_answer = re.sub(
        citation_pattern,
        check_citation,
        answer
    )

    return validated_answer, invalid_citations
