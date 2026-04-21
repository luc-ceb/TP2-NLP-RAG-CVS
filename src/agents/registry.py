"""
registry.py
-----------
Catálogo de agentes. Deriva las definiciones desde `config.AGENTS` para que
agregar un nuevo candidato solo requiera actualizar config.py.

Cada agente expone:
    slug         - clave única lowercase        (ej. "cv1")
    display_name - nombre legible para UI/LLM   (ej. "Candidato 1")
    pdf          - filename del CV en CVS_DIR   (ej. "cv1.pdf")
    aliases      - patrones regex para routing  (usados con re.search)
    namespace    - namespace en Pinecone        (ej. "cv_cv1")
"""

from src import config


def get_all_slugs() -> list[str]:
    """Devuelve todos los slugs registrados."""
    return list(config.AGENTS.keys())


def get_display_name(slug: str) -> str:
    """Devuelve el nombre legible del agente."""
    return config.AGENTS[slug]["display_name"]


def get_pdf_filename(slug: str) -> str:
    """Devuelve el filename del PDF (sin el path de CVS_DIR)."""
    return config.AGENTS[slug]["pdf"]


def get_aliases(slug: str) -> list[str]:
    """Devuelve los patrones regex usados por el router."""
    return config.AGENTS[slug].get("aliases", [])


def get_namespace(slug: str) -> str:
    """Devuelve el namespace de Pinecone para el agente."""
    return f"{config.PINECONE_NAMESPACE_PREFIX}_{slug}"


def get_default_slug() -> str:
    """Slug usado cuando la query no menciona a nadie."""
    return config.DEFAULT_AGENT
