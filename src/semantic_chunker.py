"""
semantic_chunker.py
-------------------
Chunking consciente del contexto usando SemanticChunker de LangChain con los
mismos embeddings locales (MiniLM multilingüe) que usa el retriever.

A diferencia de RecursiveCharacterTextSplitter, que corta por cuenta de
caracteres, SemanticChunker detecta límites semánticos naturales calculando
la similitud coseno entre oraciones consecutivas y corta donde la similitud
cae significativamente (cambio de tópico).

Usamos el MISMO modelo de embeddings que el retriever para que el chunking
sea coherente con el espacio vectorial donde después se busca.

Estrategia 'percentile': corta en el top N% de caídas de similitud. Un
threshold de 85 corta solo en el 15% de transiciones más abruptas.
"""

from typing import List

from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings

from src import config

_BREAKPOINT_STRATEGY = "percentile"
_BREAKPOINT_THRESHOLD = 85


def build_semantic_chunker() -> SemanticChunker:
    """Construye un SemanticChunker con los embeddings locales."""
    embeddings = HuggingFaceEmbeddings(
        model_name=config.EMBED_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )
    return SemanticChunker(
        embeddings,
        breakpoint_threshold_type=_BREAKPOINT_STRATEGY,
        breakpoint_threshold_amount=_BREAKPOINT_THRESHOLD,
    )


def semantic_chunk(pages: List[Document]) -> List[Document]:
    """
    Divide las páginas de UN CV en chunks semánticamente coherentes.
    Preserva la metadata 'source'/'agent'/'author' del primer documento.

    Parameters
    ----------
    pages : list[Document]
        Páginas ya limpias provenientes de PyPDFLoader, todas del mismo CV.

    Returns
    -------
    list[Document]
    """
    if not pages:
        return []

    chunker = build_semantic_chunker()
    full_text = "\n\n".join(p.page_content for p in pages)
    chunks = chunker.create_documents([full_text])

    # Propagamos la metadata de la primera página a todos los chunks
    base_meta = dict(pages[0].metadata)
    for chunk in chunks:
        chunk.metadata.update(base_meta)
        chunk.metadata["chunking_method"] = "semantic_minilm"

    return chunks
