
from backend.llm import generate_answer


question = "What is Artificial Intelligence?"

context = """
Source: sample.pdf
Page: 21

Content:
Artificial Intelligence is a branch of computer science
that focuses on creating machines capable of performing
tasks that normally require human intelligence.

-------------------------
"""


if __name__ == "__main__":

    print("Sending request to DocuMind AI...", flush=True)

    answer = generate_answer(question, context)

    print("\nDocuMind AI Response:")
    print(answer)
