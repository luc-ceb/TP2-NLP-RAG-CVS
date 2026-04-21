"""
config.py
---------
Centralizes all configuration values for the multi-agent RAG pipeline.
Loads environment variables from .env and exposes typed constants.

All tuneable parameters live here. Other modules should never
construct paths or read os.environ directly.
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
# Un único índice; cada agente usa su propio namespace ("cv_{slug}").
PINECONE_INDEX_NAME: str = os.getenv("PINECONE_INDEX_NAME", "cv-rag")
PINECONE_NAMESPACE_PREFIX: str = os.getenv("PINECONE_NAMESPACE_PREFIX", "cv")
PINECONE_CLOUD: str = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION: str = os.getenv("PINECONE_REGION", "us-east-1")
PINECONE_DIMENSION: int = 384          # MiniLM multilingual -> 384 dims
PINECONE_METRIC: str = "cosine"

# --- Embedding model (runs locally via sentence-transformers) ---
EMBED_MODEL: str = os.getenv(
    "EMBED_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)

# --- Groq LLM ---
GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
GROQ_TEMPERATURE: float = float(os.getenv("GROQ_TEMPERATURE", "0.2"))

# --- RAG parameters ---
TOP_K: int = int(os.getenv("TOP_K", "4"))
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "50"))

# --- Data paths ---
CVS_DIR: Path = Path(os.getenv("CVS_DIR", str(PROJECT_ROOT / "data" / "cvs")))


# ---------------------------------------------------------------------------
# Multi-agent configuration
# ---------------------------------------------------------------------------
# Cada entrada define un agente ligado a un CV puntual. El slug es la clave
# canónica (lowercase, sin espacios) y se usa como sufijo de namespace.
#
# Campos:
#   display_name : nombre legible para UI y prompts
#   pdf          : filename del CV dentro de CVS_DIR
#   aliases      : lista de expresiones regulares (re.IGNORECASE) que se
#                  evalúan contra la query en route_node para decidir qué
#                  agente(s) deben responder. Se usan con re.search().
#
# IMPORTANTE: actualizá los `display_name` y `aliases` al nombre real de cada
# persona una vez que conozcas el contenido de cada CV (ej. "Ana García" con
# aliases [r"\bana\b", r"\bana\s*garcía\b"]).
AGENTS: dict = {
    "cv1": {
        "display_name": "Luciano Ceballos",
        "pdf": "cv1.pdf",
        "aliases": [
            r"\bluciano\b",
            r"\bceballos\b",
            r"\bluciano\s+ceballos\b",
        ],
    },
    "cv2": {
        "display_name": "Matías Ignacio Rossi",
        "pdf": "cv2.pdf",
        "aliases": [
            r"\bmat[ií]as\b",
            r"\brossi\b",
            r"\bmat[ií]as\s+(?:ignacio\s+)?rossi\b",
        ],
    },
    "cv3": {
        "display_name": "Valeria Sofía Domínguez",
        "pdf": "cv3.pdf",
        "aliases": [
            r"\bvaleria\b",
            r"\bdom[ií]nguez\b",
            r"\bvaleria\s+(?:sof[ií]a\s+)?dom[ií]nguez\b",
        ],
    },
    "cv4": {
        "display_name": "Daniel Alejandro Méndez",
        "pdf": "cv4.pdf",
        "aliases": [
            r"\bdaniel\b",
            r"\bm[eé]ndez\b",
            r"\bdaniel\s+(?:alejandro\s+)?m[eé]ndez\b",
        ],
    },
}
DEFAULT_AGENT: str = os.getenv("DEFAULT_AGENT", "cv1")

# Slug por defecto cuando la query no menciona a nadie explícitamente.
DEFAULT_AGENT: str = os.getenv("DEFAULT_AGENT", "cv1")
