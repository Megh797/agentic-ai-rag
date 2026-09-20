from .config import (
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    PINECONE_NAMESPACE,
    TOP_K,
)

from .embeddings import get_embeddings

from pinecone import Pinecone


# Create Pinecone client
pc = Pinecone(
    api_key=PINECONE_API_KEY
)

# Connect to existing index
index = pc.Index(
    PINECONE_INDEX_NAME
)


def retrieve(question: str):
    """
    Retrieve the most relevant chunks from Pinecone.
    """

    embeddings = get_embeddings()

    query_vector = embeddings.embed_query(
        question
    )

    results = index.query(
        vector=query_vector,
        top_k=TOP_K,
        namespace=PINECONE_NAMESPACE,
        include_metadata=True,
    )

    contexts = []

    for match in results.get("matches", []):

        metadata = match.get(
            "metadata",
            {}
        )

        contexts.append(
            {
                "id": match.get("id"),
                "score": float(
                    match.get("score", 0.0)
                ),
                "text": metadata.get(
                    "text",
                    ""
                ),
                "page": metadata.get(
                    "page"
                ),
                "source": metadata.get(
                    "source",
                    "Ebook-Agentic-AI.pdf"
                ),
            }
        )

    return contexts