from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec

from .config import (
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    PINECONE_NAMESPACE,
    PINECONE_CLOUD,
    PINECONE_REGION,
    EMBEDDING_DIMENSION,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    PDF_PATH,
)

from .embeddings import get_embeddings


def get_pinecone():
    """Create Pinecone client."""

    return Pinecone(
        api_key=PINECONE_API_KEY
    )


def ensure_index():
    """Create the Pinecone index if it doesn't exist."""

    pc = get_pinecone()

    existing_indexes = [
        item["name"]
        for item in pc.list_indexes()
    ]

    if PINECONE_INDEX_NAME not in existing_indexes:
        print(
            f"Creating Pinecone index: "
            f"{PINECONE_INDEX_NAME}"
        )

        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(
                cloud=PINECONE_CLOUD,
                region=PINECONE_REGION,
            ),
        )

    return pc.Index(
        PINECONE_INDEX_NAME
    )


def load_pdf():
    """Load the Agentic AI ebook."""

    pdf_path = Path(PDF_PATH)

    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF not found: {pdf_path}"
        )

    print(
        f"Loading PDF: {pdf_path}"
    )

    loader = PyPDFLoader(
        str(pdf_path)
    )

    documents = loader.load()

    print(
        f"Loaded {len(documents)} pages."
    )

    return documents


def split_documents(documents):
    """Split PDF text into RAG chunks."""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            "",
        ],
    )

    chunks = splitter.split_documents(
        documents
    )

    for index, chunk in enumerate(chunks):

        chunk.metadata["chunk_id"] = (
            f"chunk-{index}"
        )

        chunk.metadata["source"] = (
            "Ebook-Agentic-AI.pdf"
        )

        chunk.metadata["page"] = (
            chunk.metadata.get(
                "page",
                0
            ) + 1
        )

    print(
        f"Created {len(chunks)} chunks."
    )

    return chunks


def ingest():
    """Run the complete PDF ingestion pipeline."""

    print("=" * 60)
    print("Agentic AI RAG - PDF Ingestion")
    print("=" * 60)

    index = ensure_index()

    documents = load_pdf()

    chunks = split_documents(
        documents
    )

    embeddings = get_embeddings()

    print(
        "Generating embeddings..."
    )

    texts = [
        chunk.page_content
        for chunk in chunks
    ]

    vectors = embeddings.embed_documents(
        texts
    )

    print(
        f"Generated {len(vectors)} embeddings."
    )

    records = []

    for chunk, vector in zip(
        chunks,
        vectors
    ):

        metadata = {
            "text": chunk.page_content,
            "source": chunk.metadata.get(
                "source",
                "Ebook-Agentic-AI.pdf"
            ),
            "page": chunk.metadata.get(
                "page",
                0
            ),
            "chunk_id": chunk.metadata.get(
                "chunk_id"
            ),
        }

        records.append(
            {
                "id": chunk.metadata[
                    "chunk_id"
                ],
                "values": vector,
                "metadata": metadata,
            }
        )

    print(
        "Uploading vectors to Pinecone..."
    )

    batch_size = 100

    for start in range(
        0,
        len(records),
        batch_size
    ):

        batch = records[
            start:start + batch_size
        ]

        index.upsert(
            vectors=batch,
            namespace=PINECONE_NAMESPACE,
        )

        print(
            f"Uploaded "
            f"{min(start + batch_size, len(records))}"
            f"/{len(records)}"
        )

    print("=" * 60)
    print("INGESTION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    ingest()