"""
orchestrator.py
---------------
API pública del sistema multi-agente. Delega en el StateGraph compilado de
graph.py. El código externo (app.py, notebooks, scripts de evaluación) solo
necesita importar `run` y `preload_agents` desde acá.

    from src.agents.orchestrator import run, preload_agents
"""

import logging
from typing import List, Optional

from langchain_core.messages import BaseMessage

from src import config
from src.agents.graph import _get_agent, rag_graph
from src.agents.person_agent import RAGResult

logger = logging.getLogger(__name__)


def run(
    question: str,
    chat_history: Optional[List[BaseMessage]] = None,
) -> List[RAGResult]:
    """
    Invoca el StateGraph compilado con la pregunta y el historial.

    El grafo se encarga de:
      1. route_node       - re.search() para detectar agente(s).
      2. conditional_edge - re.match() para elegir rama.
      3. rag_single_node  - 1 agente, o
         rag_multi_node + synthesize_node  - N agentes en paralelo + síntesis.

    Returns
    -------
    list[RAGResult]
        Siempre contiene al menos un resultado:
        - 1 RAGResult cuando respondió un solo agente
        - 1 RAGResult de síntesis cuando respondieron varios
    """
    if chat_history is None:
        chat_history = []

    output = rag_graph.invoke({
        "question": question,
        "chat_history": chat_history,
        "target_slugs": [],
        "route_type": "",
        "individual_results": [],
        "final_results": [],
    })
    return output["final_results"]


def preload_agents() -> None:
    """
    Pre-construye todos los agentes registrados (modelo de embeddings +
    conexión Pinecone). Llamar una vez al arrancar la app para evitar
    latencia en la primera query.
    """
    for slug in config.AGENTS:
        _get_agent(slug)
