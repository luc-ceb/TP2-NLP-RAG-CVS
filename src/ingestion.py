"""
ingestion.py
------------
Script de ingesta multi-agente. Cada CV se indexa en SU PROPIO namespace
de Pinecone ("cv_{slug}"), de forma que el retrieval esté aislado por persona.

Pipeline por agente:
    data/cvs/{pdf}
        -> PyPDFLoader
        -> clean_spaced_text()
        -> RecursiveCharacterTextSplitter  (default)
           o SemanticChunker (--use-semantic)
        -> HuggingFaceEmbeddings (384 dims)
        -> PineconeVectorStore  (namespace="cv_{slug}")

Uso:
    python -m src.ingestion --agent cv1
    python -m src.ingestion --agent all
    python -m src.ingestion --agent cv1 --force
    python -m src.ingestion --agent all --use-semantic
"""

import argparse
import logging
import re
import sys
from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from src import config
from src.agents.registry import get_display_name, get_namespace, get_pdf_filename

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Limpieza de artefactos de extracción PDF (idéntica a TP2)
# ---------------------------------------------------------------------------

_ACCENT_FIX_MAP = {
    "a´": "á", "e´": "é", "i´": "í", "o´": "ó", "u´": "ú",
    "A´": "Á", "E´": "É", "I´": "Í", "O´": "Ó", "U´": "Ú",
    "n~": "ñ", "N~": "Ñ",
}


def _fix_accents(text: str) -> str:
    text = re.sub(r"([´`¨~])\s+([aeiouAEIOUn])", r"\1\2", text)
    text = re.sub(r"´ı", "í", text)
    text = re.sub(r"´I", "Í", text)
    for wrong, right in _ACCENT_FIX_MAP.items():
        text = text.replace(wrong, right)
    text = re.sub(
        r"´([aeiouAEIOU])",
        lambda m: {"a": "á", "e": "é", "i": "í", "o": "ó", "u": "ú",
                   "A": "Á", "E": "É", "I": "Í", "O": "Ó", "U": "Ú"}[m.group(1)],
        text,
    )
    return text


def _fix_spaced_line(line: str) -> str:
    stripped = line.strip()
    if not stripped:
        return line
    tokens = stripped.split(" ")
    single = sum(1 for t in tokens if len(t) == 1 and t.isalnum())
    if len(tokens) > 2 and single / len(tokens) >= 0.5:
        cleaned = re.sub(r"(?<=[^\s]) (?=[^\s])", "", line)
        return re.sub(r" {2,}", " ", cleaned)
    return line


def clean_spaced_text(text: str) -> str:
    """Normaliza artefactos comunes de PyPDF: letter-spacing y acentos sueltos."""
    text = "\n".join(_fix_spaced_line(ln) for ln in text.split("\n"))
    text = _fix_accents(text)
    return text


def _clean_documents(docs: List[Document]) -> List[Document]:
    for doc in docs:
        doc.page_content = clean_spaced_text(doc.page_content)
    return docs


# ---------------------------------------------------------------------------
# Pinecone helpers
# ---------------------------------------------------------------------------

def _ensure_index(pc: Pinecone) -> None:
    """Crea el índice de Pinecone si no existe."""
    existing = [idx.name for idx in pc.list_indexes()]
    if config.PINECONE_INDEX_NAME not in existing:
        logger.info("Creando índice Pinecone '%s'...", config.PINECONE_INDEX_NAME)
        pc.create_index(
            name=config.PINECONE_INDEX_NAME,
            dimension=config.PINECONE_DIMENSION,
            metric=config.PINECONE_METRIC,
            spec=ServerlessSpec(
                cloud=config.PINECONE_CLOUD,
                region=config.PINECONE_REGION,
            ),
        )
        logger.info("Índice creado.")
    else:
        logger.info("El índice '%s' ya existe.", config.PINECONE_INDEX_NAME)


# ---------------------------------------------------------------------------
# Ingesta por agente
# ---------------------------------------------------------------------------

def ingest(
    agent_slug: str,
    force: bool = False,
    use_semantic: bool = False,
) -> int:
    """
    Indexa el CV de UN agente en su namespace propio.

    Parameters
    ----------
    agent_slug : str
        Clave de config.AGENTS (ej. "cv1").
    force : bool
        Si True, borra los vectores del namespace antes de reindexar.
    use_semantic : bool
        Si True, usa SemanticChunker en lugar de RecursiveCharacterTextSplitter.

    Returns
    -------
    int
        Número de chunks indexados.
    """
    if agent_slug not in config.AGENTS:
        raise ValueError(
            f"Agente desconocido '{agent_slug}'. "
            f"Disponibles: {list(config.AGENTS.keys())}"
        )

    display_name = get_display_name(agent_slug)
    pdf_filename = get_pdf_filename(agent_slug)
    pdf_path = config.CVS_DIR / pdf_filename
    namespace = get_namespace(agent_slug)

    if not pdf_path.is_file():
        raise FileNotFoundError(
            f"CV no encontrado para {display_name} en: {pdf_path}\n"
            f"Copiá el PDF en {config.CVS_DIR} con nombre '{pdf_filename}' y reintentá."
        )

    # 1. Cargar PDF
    logger.info("[%s] Cargando %s", display_name, pdf_path)
    loader = PyPDFLoader(str(pdf_path))
    pages = loader.load()
    logger.info("[%s] %d página(s) cargadas.", display_name, len(pages))

    # 2. Normalizar metadata + limpieza de texto
    for page in pages:
        page.metadata["source"] = pdf_path.name
        page.metadata["agent"] = agent_slug
        page.metadata["author"] = display_name
    pages = _clean_documents(pages)

    # 3. Split en chunks
    if use_semantic:
        logger.info("[%s] Chunking: SemanticChunker (MiniLM).", display_name)
        from src.semantic_chunker import semantic_chunk
        chunks = semantic_chunk(pages)
    else:
        logger.info(
            "[%s] Chunking: Recursive (size=%d, overlap=%d).",
            display_name, config.CHUNK_SIZE, config.CHUNK_OVERLAP,
        )
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        chunks = splitter.split_documents(pages)

    # Reforzamos metadata en todos los chunks (algunos splitters la pierden)
    for chunk in chunks:
        chunk.metadata.setdefault("source", pdf_path.name)
        chunk.metadata["agent"] = agent_slug
        chunk.metadata["author"] = display_name
    logger.info("[%s] %d chunks generados.", display_name, len(chunks))

    # 4. Embeddings
    logger.info("Cargando modelo de embeddings: %s", config.EMBED_MODEL)
    embeddings = HuggingFaceEmbeddings(
        model_name=config.EMBED_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )

    # 5. Asegurar que el índice exista
    pc = Pinecone(api_key=config.PINECONE_API_KEY)
    _ensure_index(pc)

    # 6. Limpiar namespace si --force
    if force:
        logger.info(
            "[%s] --force: borrando vectores del namespace '%s'...",
            display_name, namespace,
        )
        try:
            pc.Index(config.PINECONE_INDEX_NAME).delete(
                delete_all=True, namespace=namespace
            )
        except Exception:
            # El namespace puede no existir todavía — está bien
            pass

    # 7. Upsert
    logger.info("[%s] Upserting %d chunks a Pinecone (ns='%s')...",
                display_name, len(chunks), namespace)
    PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=config.PINECONE_INDEX_NAME,
        pinecone_api_key=config.PINECONE_API_KEY,
        namespace=namespace,
    )
    logger.info("[%s] Ingesta completa. %d chunks indexados.", display_name, len(chunks))
    return len(chunks)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingesta multi-agente de CVs a Pinecone.")
    parser.add_argument(
        "--agent",
        required=True,
        choices=list(config.AGENTS.keys()) + ["all"],
        help="Slug del agente a indexar, o 'all' para indexar todos.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Borrar el namespace antes de reindexar (evita duplicados).",
    )
    parser.add_argument(
        "--use-semantic",
        action="store_true",
        dest="use_semantic",
        help="Usar SemanticChunker en lugar de RecursiveCharacterTextSplitter.",
    )
    args = parser.parse_args()

    targets = list(config.AGENTS.keys()) if args.agent == "all" else [args.agent]
    failed = False
    for slug in targets:
        try:
            total = ingest(slug, force=args.force, use_semantic=args.use_semantic)
            print(f"  ✓ {slug}: {total} chunks indexados.")
        except FileNotFoundError as exc:
            logger.error(str(exc))
            failed = True
        except Exception as exc:
            logger.error("[%s] Error inesperado: %s", slug, exc)
            failed = True

    sys.exit(1 if failed else 0)
