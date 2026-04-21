"""
ingestion.py
------------
Ingesta de múltiples CVs en formato PDF a un índice de Pinecone.

Pipeline (default):
    data/cvs/*.pdf
        -> PyPDFLoader                     (extract text pages por archivo)
        -> clean_spaced_text()             (fix PDF letter-spacing artifact)
        -> RecursiveCharacterTextSplitter  (split por carácter)
        -> HuggingFaceEmbeddings           (embed local)
        -> PineconeVectorStore             (upsert)

Pipeline (--use-semantic):
    data/cvs/*.pdf
        -> PyPDFLoader
        -> clean_spaced_text()
        -> SemanticChunker + HuggingFace MiniLM  (split por tópico)
        -> HuggingFaceEmbeddings
        -> PineconeVectorStore

Uso:
    python -m src.ingestion [--cvs-dir PATH] [--force] [--use-semantic]
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# Mapa de acentos sueltos (combining marks + ASCII) a letras con tilde
_ACCENT_FIX_MAP = {
    "a´": "á", "e´": "é", "i´": "í", "o´": "ó", "u´": "ú",
    "A´": "Á", "E´": "É", "I´": "Í", "O´": "Ó", "U´": "Ú",
    "n~": "ñ", "N~": "Ñ",
    "a`": "à", "e`": "è", "i`": "ì", "o`": "ò", "u`": "ù",
    "a¨": "ä", "e¨": "ë", "i¨": "ï", "o¨": "ö", "u¨": "ü",
    # Versión con espacio: "Dom´ ınguez" -> "Domínguez"
    # La í/ı sola después del acento se debe al dotless-i del PDF
}


def _fix_accents(text: str) -> str:
    """
    Corrige acentos extraídos como carácter separado.
    Casos cubiertos:
      'Dom´ınguez' -> 'Domínguez'  (acento + i sin punto)
      'Dom´ ınguez' -> 'Domínguez' (acento + espacio + i sin punto)
      'Tel´efono' -> 'Teléfono'
    """
    # Paso 1: eliminar espacios entre acento y letra siguiente
    # 'Dom´ ınguez' -> 'Dom´ınguez'
    text = re.sub(r"([´`¨~])\s+([aeiouAEIOUn])", r"\1\2", text)
    # Paso 2: convertir 'ı' (dotless i, común en PDFs) + acento previo a 'í'
    text = re.sub(r"´ı", "í", text)
    text = re.sub(r"´I", "Í", text)
    # Paso 3: mapear las combinaciones letra+acento a la letra con tilde
    for wrong, right in _ACCENT_FIX_MAP.items():
        text = text.replace(wrong, right)
    # Paso 4: acento+letra en orden PDF (´a -> á)
    text = re.sub(r"´([aeiouAEIOU])",
                  lambda m: {"a":"á","e":"é","i":"í","o":"ó","u":"ú",
                             "A":"Á","E":"É","I":"Í","O":"Ó","U":"Ú"}[m.group(1)],
                  text)
    return text


def _fix_spaced_line(line: str) -> str:
    """Corrige líneas donde la extracción del PDF separó cada carácter con un espacio."""
    stripped = line.strip()
    if not stripped:
        return line
    tokens = stripped.split(" ")
    single = sum(1 for t in tokens if len(t) == 1 and t.isalnum())
    if len(tokens) > 2 and single / len(tokens) >= 0.5:
        cleaned = re.sub(r"(?<=[^\s]) (?=[^\s])", "", line)
        cleaned = re.sub(r" {2,}", " ", cleaned)
        return cleaned
    return line


def clean_spaced_text(text: str) -> str:
    """
    Normaliza artefactos típicos de extracción de PDFs:
      - Letter-spacing: 's t r u c t u r e' -> 'structure'
      - Acentos sueltos: 'Dom´ ınguez' -> 'Domínguez'
    """
    # Primero fix de espaciado, después fix de acentos
    text = "\n".join(_fix_spaced_line(ln) for ln in text.split("\n"))
    text = _fix_accents(text)
    return text

def _clean_documents(docs: List[Document]) -> List[Document]:
    """Aplica clean_spaced_text a todos los documentos in-place."""
    for doc in docs:
        doc.page_content = clean_spaced_text(doc.page_content)
    return docs


# ---------------------------------------------------------------------------
# Carga multi-PDF
# ---------------------------------------------------------------------------

def _load_all_pdfs(cvs_dir: Path) -> List[Document]:
    """
    Carga todos los PDFs de la carpeta y normaliza la metadata 'source' al
    nombre de archivo (sin path absoluto). Esto permite citar fuentes como
    [cv_juan_perez.pdf] en las respuestas del LLM.
    """
    pdf_paths = sorted(cvs_dir.glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(f"No se encontraron PDFs en {cvs_dir}")

    logger.info("Encontrados %d PDFs en %s", len(pdf_paths), cvs_dir)
    all_docs: List[Document] = []
    for pdf in pdf_paths:
        loader = PyPDFLoader(str(pdf))
        pages = loader.load()
        # Normalizar source al filename
        for page in pages:
            page.metadata["source"] = pdf.name
            page.metadata["cv_file"] = pdf.name  # alias explícito
        logger.info("  %s: %d página(s)", pdf.name, len(pages))
        all_docs.extend(pages)

    logger.info("Total: %d página(s) cargadas.", len(all_docs))
    return all_docs


def _build_embeddings() -> HuggingFaceEmbeddings:
    """Inicializa el modelo de embeddings local."""
    logger.info("Cargando modelo de embeddings: %s", config.EMBED_MODEL)
    return HuggingFaceEmbeddings(
        model_name=config.EMBED_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )


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
# Función principal
# ---------------------------------------------------------------------------

def ingest(cvs_dir: str, force: bool = False, use_semantic: bool = False) -> int:
    """
    Pipeline completo: load PDFs → clean → chunk → embed → upsert Pinecone.

    Parameters
    ----------
    cvs_dir : str
        Directorio con los PDFs de CVs.
    force : bool
        Si True, borra los vectores existentes del namespace antes de reindexar.
    use_semantic : bool
        Si True, usa SemanticChunker (MiniLM) en lugar de splitting por caracteres.

    Returns
    -------
    int
        Número de chunks indexados.
    """
    cvs_path = Path(cvs_dir)
    if not cvs_path.is_dir():
        raise FileNotFoundError(f"Directorio no encontrado: {cvs_dir}")

    # 1. Cargar todos los PDFs
    pages = _load_all_pdfs(cvs_path)

    # 2. Limpiar artefactos de extracción
    pages = _clean_documents(pages)
    logger.info("Limpieza de texto aplicada.")

    # 3. Split en chunks
    if use_semantic:
        logger.info("Estrategia de chunking: SemanticChunker (HuggingFace MiniLM)")
        from src.semantic_chunker import semantic_chunk
        chunks = semantic_chunk(pages)
    else:
        logger.info(
            "Estrategia de chunking: RecursiveCharacterTextSplitter (size=%d, overlap=%d)",
            config.CHUNK_SIZE, config.CHUNK_OVERLAP,
        )
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        chunks = splitter.split_documents(pages)
    logger.info("Total de chunks generados: %d", len(chunks))

    # 4. Embeddings
    embeddings = _build_embeddings()

    # 5. Crear índice si hace falta
    pc = Pinecone(api_key=config.PINECONE_API_KEY)
    _ensure_index(pc)

    # 6. Limpiar vectores existentes opcionalmente
    if force:
        logger.info(
            "--force: borrando vectores existentes en '%s' (namespace='%s')...",
            config.PINECONE_INDEX_NAME, config.PINECONE_NAMESPACE,
        )
        try:
            pc.Index(config.PINECONE_INDEX_NAME).delete(
                delete_all=True, namespace=config.PINECONE_NAMESPACE
            )
        except Exception:
            pass  # namespace aún no existe
        logger.info("Índice vaciado.")

    # 7. Upsert
    logger.info("Upserting %d chunks a Pinecone...", len(chunks))
    PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=config.PINECONE_INDEX_NAME,
        pinecone_api_key=config.PINECONE_API_KEY,
        namespace=config.PINECONE_NAMESPACE,
    )
    logger.info("Ingesta completa. %d chunks indexados.", len(chunks))
    return len(chunks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingesta de CVs en PDF a Pinecone.")
    parser.add_argument(
        "--cvs-dir",
        default=str(config.CVS_DIR),
        help=f"Directorio con PDFs (default: {config.CVS_DIR})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Borrar vectores existentes antes de indexar (evita duplicados).",
    )
    parser.add_argument(
        "--use-semantic",
        action="store_true",
        dest="use_semantic",
        help="Usar SemanticChunker (HuggingFace MiniLM) en lugar de RecursiveCharacterTextSplitter.",
    )
    args = parser.parse_args()

    try:
        total = ingest(args.cvs_dir, force=args.force, use_semantic=args.use_semantic)
        sys.exit(0)
    except FileNotFoundError as exc:
        logger.error(str(exc))
        logger.error("Colocá tus CVs en data/cvs/ y volvé a correr.")
        sys.exit(1)
