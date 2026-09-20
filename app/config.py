import os
from dotenv import load_dotenv

load_dotenv()

# Pinecone
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

PINECONE_INDEX_NAME = os.getenv(
    "PINECONE_INDEX_NAME",
    "agentic-ai-ebook"
)

PINECONE_NAMESPACE = os.getenv(
    "PINECONE_NAMESPACE",
    "agentic-ai-ebook"
)

PINECONE_CLOUD = os.getenv(
    "PINECONE_CLOUD",
    "aws"
)

PINECONE_REGION = os.getenv(
    "PINECONE_REGION",
    "us-east-1"
)

# RAG
CHUNK_SIZE = int(
    os.getenv("CHUNK_SIZE", "900")
)

CHUNK_OVERLAP = int(
    os.getenv("CHUNK_OVERLAP", "150")
)

TOP_K = int(
    os.getenv("TOP_K", "5")
)

MIN_RELEVANCE_SCORE = float(
    os.getenv("MIN_RELEVANCE_SCORE", "0.45")
)

# Embedding model
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

EMBEDDING_DIMENSION = 384

# PDF
PDF_PATH = "data/Ebook-Agentic-AI.pdf"

