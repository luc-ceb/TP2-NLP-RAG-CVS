"""
config.py
---------
Centraliza todos los valores de configuracion para el pipeline del RAG
Carga todas las variables de entorno desde .env y exporta las constantes typeadas
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(override=True)

# Absolute path to the project root (one level above src/)
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent


def _require(key: str) -> str:
    """Retrieve a required environment variable or raise."""
    value = os.getenv(key, "").strip()
    if not value:
        raise EnvironmentError(f"Required environment variable '{key}' is not set.")
    return value


# --- API credentials ---
GROQ_API_KEY: str = _require("GROQ_API_KEY")
PINECONE_API_KEY: str = _require("PINECONE_API_KEY")

# --- Pinecone index ---
PINECONE_INDEX_NAME: str = os.getenv("PINECONE_INDEX_NAME", "cv-rag")
PINECONE_NAMESPACE: str = os.getenv("PINECONE_NAMESPACE", "recursive")
PINECONE_CLOUD: str = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION: str = os.getenv("PINECONE_REGION", "us-east-1")
PINECONE_DIMENSION: int = 384          # Dimension of all-MiniLM-L6-v2 embeddings
PINECONE_METRIC: str = "cosine"

# --- Embedding model (runs locally via sentence-transformers) ---
# Para español multilingüe, paraphrase-multilingual-MiniLM-L12-v2 funciona mejor.
# Para inglés, all-MiniLM-L6-v2 es más rápido. Ambos devuelven 384 dims.
EMBED_MODEL: str = os.getenv("EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# --- Groq LLM ---
GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
GROQ_TEMPERATURE: float = float(os.getenv("GROQ_TEMPERATURE", "0.2"))

# --- RAG parameters ---
TOP_K: int = int(os.getenv("TOP_K", "4"))
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "50"))

# --- Data paths ---
# Carpeta con múltiples PDFs de CVs. Cada archivo se indexa con su filename
# como metadata "source" para permitir citar fuentes en las respuestas.
CVS_DIR: Path = Path(os.getenv("CVS_DIR", str(PROJECT_ROOT / "data" / "cvs")))
