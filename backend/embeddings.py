
from functools import lru_cache

from sentence_transformers import SentenceTransformer


@lru_cache(maxsize=1)
def load_embedding_model():
    """Load the embedding model only when first required."""

    print("Loading embedding model...", flush=True)

    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("Embedding model loaded successfully!", flush=True)

    return model


def generate_embeddings(chunks):
    """Generate embeddings for document chunks."""

    if not chunks:
        return []

    model = load_embedding_model()

    texts = [
        chunk["text"] for chunk in chunks
    ]

    embeddings = model.encode(texts)

    return embeddings


if __name__ == "__main__":

    sample_chunks = [
        {"text": "Artificial Intelligence is transforming industries."},
        {"text": "Machine Learning allows computers to learn from data."},
        {"text": "Deep Learning uses neural networks."}
    ]

    vectors = generate_embeddings(sample_chunks)

    print("Total embeddings:", len(vectors))
    print("Vector dimensions:", vectors.shape)
    print("First embedding:", vectors[0])
