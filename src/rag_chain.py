"""
rag_chain.py
------------
Shim de compatibilidad. La lógica real vive en `src.agents.orchestrator`.

Mantiene esta ruta de import estable para notebooks y scripts viejos.
Reexporta la API pública del sistema multi-agente.
"""

from src.agents.orchestrator import run, preload_agents
from src.agents.person_agent import RAGResult

__all__ = ["run", "preload_agents", "RAGResult"]
