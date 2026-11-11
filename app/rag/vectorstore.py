import hashlib
import time
from pathlib import Path

from pinecone import Pinecone, ServerlessSpec
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore

from app.core.config import get_settings


settings = get_settings()

_embeddings = None
_vectorstore = None

EMBEDDING_DIMENSIONS = {
    "models/gemini-embedding-001": 3072,
    "gemini-embedding-001": 3072,
}


def get_embedding_dimension(model_name: str | None = None) -> int:
    name = (model_name or settings.embedding_model or "").strip()

    if not name:
        raise RuntimeError("Embedding model is not configured")

    normalized = name.lower()

    if normalized in EMBEDDING_DIMENSIONS:
        return EMBEDDING_DIMENSIONS[normalized]

    raise ValueError(
        f"Unsupported embedding model '{name}'. "
        "Add its matching dimension to EMBEDDING_DIMENSIONS."
    )


def get_embeddings():
    global _embeddings

    if _embeddings is None:
        if not settings.gemini_api_key:
            raise RuntimeError("GEMINI_API_KEY is missing")

        _embeddings = GoogleGenerativeAIEmbeddings(
            model=settings.embedding_model,
            google_api_key=settings.gemini_api_key,
        )

    return _embeddings


def ensure_index():

    if not settings.pinecone_api_key:
        raise RuntimeError("PINECONE_API_KEY is missing")

    desired_dimension = get_embedding_dimension()

    pc = Pinecone(api_key=settings.pinecone_api_key)

    index_names = pc.list_indexes().names()

    if settings.pinecone_index_name in index_names:

        index_info = pc.describe_index(
            settings.pinecone_index_name
        )

        current_dimension = getattr(
            index_info,
            "dimension",
            None
        )

        if current_dimension != desired_dimension:
            raise RuntimeError(
                f"Pinecone index dimension mismatch. "
                f"Existing index: {current_dimension}, "
                f"Required dimension: {desired_dimension}. "
                "Create a new Pinecone index with the correct "
                "dimension or migrate your existing vectors."
            )

    else:

        pc.create_index(
            name=settings.pinecone_index_name,
            dimension=desired_dimension,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            ),
        )

        
        max_attempts = 60
        attempt = 0
        while attempt < max_attempts:
            index_info = pc.describe_index(
                settings.pinecone_index_name
            )

            status = getattr(index_info, "status", {})

            if isinstance(status, dict):
                ready = status.get("ready", False)
            else:
                ready = getattr(status, "ready", False)

            if ready:
                break

            time.sleep(1)
            attempt += 1
        else:
            raise TimeoutError(
                f"Pinecone index '{settings.pinecone_index_name}' did not become ready within {max_attempts} seconds."
            )

    return pc.Index(settings.pinecone_index_name)


def get_vectorstore():
    global _vectorstore

    if _vectorstore is None:

        index = ensure_index()

        _vectorstore = PineconeVectorStore(
            index=index,
            embedding=get_embeddings(),
            namespace=settings.pinecone_namespace,
        )

    return _vectorstore


def get_retriever():

    return get_vectorstore().as_retriever(
        search_kwargs={
            "k": settings.top_k
        }
    )


def add_documents(chunks):
    store = get_vectorstore()
    ids = []
    for idx, chunk in enumerate(chunks):
        src = Path(chunk.metadata.get("source", "doc")).name
        content_hash = hashlib.sha256(chunk.page_content.encode("utf-8")).hexdigest()[:16]
        ids.append(f"{src}_{idx}_{content_hash}")

    return store.add_documents(chunks, ids=ids)
