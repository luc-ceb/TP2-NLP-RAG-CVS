"""
graph.py
--------
StateGraph de LangGraph que orquesta el pipeline multi-agente.

Diagrama de flujo
-----------------

    __START__
        |
        v
    route_node          <- re.search() detecta slugs en la query
        |
        | conditional_edge (re.match(r'^single$', route_type))
        +-- route_type == "single" --> rag_single_node  -----------> __END__
        |
        +-- route_type == "multi"  --> rag_multi_node
                                            |
                                     synthesize_node     -----------> __END__

Nodos:
    route_node      - Decide qué agente(s) deben responder (re.search sobre aliases).
    rag_single_node - Invoca 1 PersonAgent con su namespace propio.
    rag_multi_node  - Invoca N PersonAgents en paralelo (ThreadPoolExecutor).
    synthesize_node - LLM combina respuestas individuales sin alucinar.

Conditional edge:
    _edge_decision(state) usa re.match(r'^single$', state["route_type"]) para
    elegir la rama.
"""

import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Literal

from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from src import config
from src.agents.person_agent import PersonAgent, RAGResult
from src.agents.registry import get_aliases, get_all_slugs, get_default_slug

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# State compartido entre nodos
# ---------------------------------------------------------------------------

class GraphState(TypedDict):
    """
    Estado que fluye entre nodos.

    Campos
    ------
    question           : Pregunta original del usuario.
    chat_history       : Historial LangChain (BaseMessage) para memoria conversacional.
    target_slugs       : Agentes que deben responder (lo escribe route_node).
    route_type         : "single" | "multi" (lo escribe route_node, lo lee el edge).
    individual_results : Resultado por agente (rag_single_node / rag_multi_node).
    final_results      : Lista devuelta al caller.
    """
    question: str
    chat_history: List[BaseMessage]
    target_slugs: List[str]
    route_type: str
    individual_results: List[RAGResult]
    final_results: List[RAGResult]


# ---------------------------------------------------------------------------
# Cache de agentes (reuso entre calls sin re-instanciar retrievers)
# ---------------------------------------------------------------------------

_agent_cache: dict[str, PersonAgent] = {}


def _get_agent(slug: str) -> PersonAgent:
    """Devuelve un PersonAgent ya construido, cacheándolo en el primer uso."""
    if slug not in _agent_cache:
        logger.info("Inicializando agente '%s'...", slug)
        _agent_cache[slug] = PersonAgent(slug).build()
    return _agent_cache[slug]


# ---------------------------------------------------------------------------
# Conditional edge: el "tomador de decisión" del grafo
# ---------------------------------------------------------------------------

def _edge_decision(state: GraphState) -> Literal["rag_single", "rag_multi"]:
    """
    Conditional edge que decide la rama de ejecución.

        re.match(r'^single$', route_type)  ->  "rag_single"
        otherwise                          ->  "rag_multi"
    """
    if re.match(r"^single$", state["route_type"]):
        logger.debug("Edge decision -> rag_single (slugs: %s)", state["target_slugs"])
        return "rag_single"
    logger.debug("Edge decision -> rag_multi (slugs: %s)", state["target_slugs"])
    return "rag_multi"


# ---------------------------------------------------------------------------
# Nodo: route_node
# ---------------------------------------------------------------------------

def route_node(state: GraphState) -> dict:
    """
    Detecta qué agentes deben responder usando re.search() sobre los aliases
    definidos en config.AGENTS.

    - Si no matchea ninguno -> DEFAULT_AGENT.
    - Si matchea uno        -> route_type = "single".
    - Si matchea >= 2       -> route_type = "multi".
    """
    query = state["question"]
    matched: list[str] = []
    for slug in get_all_slugs():
        for pattern in get_aliases(slug):
            if re.search(pattern, query, re.IGNORECASE):
                matched.append(slug)
                break  # un alias alcanza para contar al agente

    if not matched:
        matched = [get_default_slug()]

    route_type = "single" if len(matched) == 1 else "multi"
    logger.info("route_node -> slugs=%s  type=%s", matched, route_type)
    return {"target_slugs": matched, "route_type": route_type}


# ---------------------------------------------------------------------------
# Nodo: rag_single_node
# ---------------------------------------------------------------------------

def rag_single_node(state: GraphState) -> dict:
    """Invoca un único PersonAgent y emite el resultado final."""
    slug = state["target_slugs"][0]
    agent = _get_agent(slug)
    result = agent.invoke(state["question"], state["chat_history"])
    logger.info("rag_single_node -> %s respondió (%d chars)", slug, len(result.answer))
    return {
        "individual_results": [result],
        "final_results": [result],
    }


# ---------------------------------------------------------------------------
# Nodo: rag_multi_node
# ---------------------------------------------------------------------------

def rag_multi_node(state: GraphState) -> dict:
    """
    Invoca múltiples PersonAgents en paralelo usando ThreadPoolExecutor.
    Cada agente responde SOLO sobre su propia persona. La síntesis se hace
    después en synthesize_node.

    Nota: Groq free tier tiene rate limits (429). Para 2-4 agentes en
    paralelo no suele ser un problema; si lo fuera, bajar max_workers a 1
    serializaría las calls.
    """
    slugs = state["target_slugs"]

    def _invoke_one(slug: str) -> tuple[str, RAGResult]:
        agent = _get_agent(slug)
        return slug, agent.invoke(state["question"], state["chat_history"])

    results: dict[str, RAGResult] = {}
    with ThreadPoolExecutor(max_workers=max(len(slugs), 1)) as executor:
        futures = {executor.submit(_invoke_one, slug): slug for slug in slugs}
        for future in as_completed(futures):
            slug, result = future.result()
            results[slug] = result

    # Preservamos el orden en el que route_node los emitió
    ordered = [results[s] for s in slugs if s in results]
    logger.info("rag_multi_node -> %d agentes respondieron", len(ordered))
    return {"individual_results": ordered}


# ---------------------------------------------------------------------------
# Nodo: synthesize_node
# ---------------------------------------------------------------------------

def synthesize_node(state: GraphState) -> dict:
    """
    Toma las respuestas individuales de rag_multi_node y las combina en una
    única respuesta comparativa. El LLM recibe como contexto las respuestas
    ya extraídas por cada agente, no los chunks originales, así no inventa.
    """
    individual = state["individual_results"]

    context_block = "\n\n".join(
        f"=== Información sobre {r.display_name} (extraída de su CV) ===\n{r.answer}"
        for r in individual
    )
    all_sources = [doc for r in individual for doc in r.source_documents]
    names_label = " vs ".join(r.display_name for r in individual)

    system = (
        "Sos un asistente que responde preguntas que involucran a varias personas. "
        "Te pasan los datos extraídos individualmente del CV de cada una. "
        "Usá EXCLUSIVAMENTE la información provista; no inventes nada. "
        "Si falta información de alguna persona, indicalo explícitamente. "
        "Respondé de forma estructurada (por persona o comparación), "
        "en el mismo idioma que la pregunta."
    )
    human = (
        "Información extraída de los CVs:\n\n{context}\n\n"
        "Pregunta: {question}\n\n"
        "Entregá una respuesta completa basada únicamente en la información anterior."
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        MessagesPlaceholder("chat_history"),
        ("human", human),
    ])

    llm = ChatGroq(
        model=config.GROQ_MODEL,
        api_key=config.GROQ_API_KEY,
        temperature=config.GROQ_TEMPERATURE,
    )

    synthesis_answer = (prompt | llm | StrOutputParser()).invoke({
        "context": context_block,
        "question": state["question"],
        "chat_history": state["chat_history"],
    })

    logger.info("synthesize_node -> síntesis lista (%d chars)", len(synthesis_answer))

    synthesis_result = RAGResult(
        agent_slug="synthesis",
        display_name=f"Comparación — {names_label}",
        answer=synthesis_answer,
        source_documents=all_sources,
    )
    return {"final_results": [synthesis_result]}


# ---------------------------------------------------------------------------
# Compilación del grafo
# ---------------------------------------------------------------------------

def build_graph():
    """
    Construye y compila el StateGraph.

    Topología:
        START -> route_node -> (conditional_edge) -> rag_single_node -> END
                                                  -> rag_multi_node  -> synthesize_node -> END
    """
    graph = StateGraph(GraphState)

    # Registro de nodos
    graph.add_node("route_node", route_node)
    graph.add_node("rag_single_node", rag_single_node)
    graph.add_node("rag_multi_node", rag_multi_node)
    graph.add_node("synthesize_node", synthesize_node)

    # Edges fijos
    graph.add_edge(START, "route_node")
    graph.add_edge("rag_multi_node", "synthesize_node")
    graph.add_edge("rag_single_node", END)
    graph.add_edge("synthesize_node", END)

    # Conditional edge: el que usa re.match para elegir la rama
    graph.add_conditional_edges(
        "route_node",
        _edge_decision,
        {
            "rag_single": "rag_single_node",
            "rag_multi": "rag_multi_node",
        },
    )

    compiled = graph.compile()
    logger.info("LangGraph compilado correctamente.")
    return compiled


# Instancia única — se compila al importar el módulo
rag_graph = build_graph()
