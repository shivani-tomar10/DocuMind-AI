
import logging
from functools import lru_cache
from typing import Any

from sentence_transformers import SentenceTransformer


logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def load_embedding_model() -> SentenceTransformer:
    """Load and cache the embedding model."""

    logger.info("Loading embedding model: %s", EMBEDDING_MODEL)

    model = SentenceTransformer(EMBEDDING_MODEL)

    logger.info("Embedding model loaded successfully.")

    return model


def generate_embeddings(chunks: list[dict[str, Any]]):
    """Generate embeddings for document chunks."""

    if not chunks:
        return []

    model = load_embedding_model()

    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(
        texts,
        show_progress_bar=False
    )

    return embeddings


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    sample_chunks = [
        {"text": "Artificial Intelligence is transforming industries."},
        {"text": "Machine Learning allows computers to learn from data."},
        {"text": "Deep Learning uses neural networks."}
    ]

    vectors = generate_embeddings(sample_chunks)

    print("Total embeddings:", len(vectors))
    print("Vector dimensions:", vectors.shape)
