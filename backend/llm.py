
import os
import re

from dotenv import load_dotenv
from google import genai


# --------------------------------
# 1. Load Gemini API configuration
# --------------------------------

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env file")

client = genai.Client(api_key=api_key)


# --------------------------------
# 2. Generate Expanded Search Query
# --------------------------------

def generate_search_query(query):

    prompt = f"""
You are a search query optimization assistant
for a document-based RAG system.

Rewrite the user's question into a focused
semantic search query.

STRICT RULES:
1. Preserve the exact intent of the original question.
2. Keep the main subject and important keywords.
3. Add useful synonyms and alternative expressions.
4. Include the type of information being requested.
5. Do not introduce unrelated subtopics.
6. Do not add examples unless requested.
7. Do not broaden a definition question into a
   general overview of the subject.
8. Keep the query concise, ideally 15-25 words.
9. Return only the rewritten search query.

USER QUESTION:
{query}

FOCUSED SEARCH QUERY:
"""

    try:
        interaction = client.interactions.create(
            model="gemini-3.5-flash-lite",
            input=prompt,
            timeout=60
        )

        expanded_query = interaction.output_text.strip()

        if not expanded_query:
            return query

        return expanded_query

    except Exception as error:

        print(
            "Query expansion failed:",
            type(error).__name__,
            str(error)
        )

        return query


# --------------------------------
# 3. Generate Final RAG Answer
# --------------------------------

def generate_answer(query, context):

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

        return answer

    except Exception as error:

        print(
            "Answer generation failed:",
            type(error).__name__,
            str(error)
        )

        return "Sorry, I could not generate an answer at this time."


# --------------------------------
# 4. Validate Generated Citations
# --------------------------------

def validate_citations(answer, metadatas):

    # Step 1: Create valid source-page pairs
    valid_sources = {
        (
            str(metadata["source"]),
            str(metadata["page_number"])
        )
        for metadata in metadatas
    }

    # Step 2: Citation pattern
    citation_pattern = (
        r"\[Source:\s*(.*?),\s*Page:\s*(\d+)\]"
    )

    invalid_citations = []

    # Step 3: Validate each citation
    def check_citation(match):

        source = match.group(1).strip()
        page = match.group(2).strip()

        citation = (source, page)

        if citation in valid_sources:

            return match.group(0)

        invalid_citations.append(match.group(0))

        return "[Unverified citation]"

    # Step 4: Replace invalid citations
    validated_answer = re.sub(
        citation_pattern,
        check_citation,
        answer
    )

    # Step 5: Return results
    return validated_answer, invalid_citations
