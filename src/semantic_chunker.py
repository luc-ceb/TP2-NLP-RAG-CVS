"""
semantic_chunker.py
-------------------
Chunking consciente del contexto usando el SemanticChunker de LangChain.

A diferencia de RecursiveCharacterTextSplitter, que corta ciegamente por cuenta
de caracteres, SemanticChunker detecta límites semánticos naturales calculando
la similitud coseno entre oraciones consecutivas. Se inserta un corte donde la
similitud cae significativamente — es decir, donde cambia el tópico.

Usa el MISMO modelo de embeddings que el retriever (HuggingFace MiniLM
multilingüe), lo que tiene dos ventajas:

  1. No requiere API keys externas. Corre 100% local.
  2. Consistencia: los chunks se cortan según el mismo espacio vectorial en el
     que después se busca. Si MiniLM considera que dos oraciones hablan del
     mismo tema, quedan en el mismo chunk y el retriever las puede encontrar
     juntas. Esto reduce la fragmentación de contexto en el retrieval.

Estrategia de breakpoint: 'percentile'
    Corta en el top N% de caídas de similitud entre oraciones consecutivas.
    Un threshold de 85 significa: cortar solo en el 15% de transiciones más
    abruptas. Menor threshold -> más chunks (cortes más finos).
"""

from typing import List

from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings

from src import config

_BREAKPOINT_STRATEGY = "percentile"
_BREAKPOINT_THRESHOLD = 85


def build_semantic_chunker() -> SemanticChunker:
    """
    Construye un SemanticChunker con los mismos embeddings que el retriever.

    Usamos el modelo declarado en config.EMBED_MODEL (por default,
    paraphrase-multilingual-MiniLM-L12-v2). Es el mismo espacio vectorial
    que se usa después para buscar, lo cual mantiene coherencia entre
    chunking y retrieval.
    """
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
    Divide las páginas de CVs en chunks semánticamente coherentes.

    Procesamos un CV por vez (agrupando por metadata 'source') para preservar
    la trazabilidad al archivo origen. Mezclar todos los CVs en un único
    stream de texto rompería el retrieval por candidato.

    Parameters
    ----------
    pages : list[Document]
        Páginas ya limpias provenientes de PyPDFLoader.

    Returns
    -------
    list[Document]
        Chunks con metadata 'source', 'cv_file' y 'chunking_method'.
    """
    chunker = build_semantic_chunker()

    # Agrupamos páginas por archivo de origen
    by_source: dict = {}
    for page in pages:
        src = page.metadata.get("source", "unknown")
        by_source.setdefault(src, []).append(page)

    all_chunks: List[Document] = []
    for src, src_pages in by_source.items():
        full_text = "\n\n".join(p.page_content for p in src_pages)
        src_chunks = chunker.create_documents([full_text])
        for chunk in src_chunks:
            chunk.metadata["source"] = src
            chunk.metadata["cv_file"] = src
            chunk.metadata["chunking_method"] = "semantic_minilm"
        all_chunks.extend(src_chunks)

    return all_chunks
