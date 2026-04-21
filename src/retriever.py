"""
retriever.py
------------
Construye un VectorStoreRetriever de LangChain conectado a un índice
preexistente de Pinecone.

Se crea una sola vez y se reutiliza en todas las queries.
"""

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.vectorstores import VectorStoreRetriever

from src import config


def build_retriever(top_k: int = None, search_type: str = "similarity") -> VectorStoreRetriever:
    """
    Conecta al índice de Pinecone existente y devuelve un retriever.

    Parameters
    ----------
    top_k : int, optional
        Número de chunks a recuperar. Por defecto usa config.TOP_K.
    search_type : str
        "similarity" (default) o "mmr" para máxima relevancia marginal.

    Returns
    -------
    VectorStoreRetriever

    Raises
    ------
    Exception
        Si el índice no existe o las credenciales son inválidas.
    """
    embeddings = HuggingFaceEmbeddings(
        model_name=config.EMBED_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )

    vector_store = PineconeVectorStore(
        index_name=config.PINECONE_INDEX_NAME,
        embedding=embeddings,
        pinecone_api_key=config.PINECONE_API_KEY,
        namespace=config.PINECONE_NAMESPACE,
    )

    k = top_k if top_k is not None else config.TOP_K

    return vector_store.as_retriever(
        search_type=search_type,
        search_kwargs={"k": k},
    )


def build_vectorstore() -> PineconeVectorStore:
    """
    Devuelve el vectorstore crudo (sin wrappear en retriever).
    Útil para llamar a similarity_search_with_score() en evaluación.
    """
    embeddings = HuggingFaceEmbeddings(
        model_name=config.EMBED_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )
    return PineconeVectorStore(
        index_name=config.PINECONE_INDEX_NAME,
        embedding=embeddings,
        pinecone_api_key=config.PINECONE_API_KEY,
        namespace=config.PINECONE_NAMESPACE,
    )
