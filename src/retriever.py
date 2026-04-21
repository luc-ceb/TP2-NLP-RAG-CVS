"""
retriever.py
------------
Construye un VectorStoreRetriever de LangChain conectado al índice de Pinecone,
scopeado al namespace de UN agente. Cada agente tiene su propio namespace
("cv_{slug}"), así el retrieval siempre está aislado al CV de esa persona.
"""

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.vectorstores import VectorStoreRetriever

from src import config
from src.agents.registry import get_namespace


def _build_embeddings() -> HuggingFaceEmbeddings:
    """Devuelve el modelo de embeddings local."""
    return HuggingFaceEmbeddings(
        model_name=config.EMBED_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )


def build_retriever(
    agent_slug: str,
    top_k: int | None = None,
    search_type: str = "similarity",
) -> VectorStoreRetriever:
    """
    Construye un retriever apuntando al namespace del agente indicado.

    Parameters
    ----------
    agent_slug : str
        Identificador del agente (debe existir en config.AGENTS).
    top_k : int, optional
        Número de chunks a recuperar. Default: config.TOP_K.
    search_type : str
        "similarity" (default) o "mmr" para máxima relevancia marginal.

    Returns
    -------
    VectorStoreRetriever
    """
    if agent_slug not in config.AGENTS:
        raise ValueError(f"Agente desconocido: '{agent_slug}'")

    vector_store = PineconeVectorStore(
        index_name=config.PINECONE_INDEX_NAME,
        embedding=_build_embeddings(),
        pinecone_api_key=config.PINECONE_API_KEY,
        namespace=get_namespace(agent_slug),
    )

    k = top_k if top_k is not None else config.TOP_K

    if search_type == "mmr":
        return vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": k, "fetch_k": k * 4, "lambda_mult": 0.6},
        )
    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )


def build_vectorstore(agent_slug: str) -> PineconeVectorStore:
    """
    Devuelve el vectorstore crudo del agente (sin wrappear en retriever).
    Útil para similarity_search_with_score() en evaluación manual.
    """
    if agent_slug not in config.AGENTS:
        raise ValueError(f"Agente desconocido: '{agent_slug}'")
    return PineconeVectorStore(
        index_name=config.PINECONE_INDEX_NAME,
        embedding=_build_embeddings(),
        pinecone_api_key=config.PINECONE_API_KEY,
        namespace=get_namespace(agent_slug),
    )
