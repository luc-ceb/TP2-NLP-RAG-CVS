"""
evaluation.py
-------------
Evaluación del sistema multi-agente:
  - Retrieval: precision@k, recall@k, MRR a nivel agente esperado.
  - Routing : accuracy del route_node (¿detectó bien los slugs?).
  - Respuesta: comparación cualitativa con/sin RAG.

El ground truth de cada query indica:
    relevant_slugs : qué agente(s) debería(n) responder (ej. ["cv1", "cv3"]).

Como ahora hay un agente por persona, la "fuente relevante" es directamente
el slug. La evaluación verifica que el router seleccione los agentes correctos
y que cada agente recupere chunks del CV correcto.
"""

from dataclasses import dataclass, field
from typing import Dict, List

import pandas as pd

from src.agents.graph import rag_graph, _get_agent
from src.agents.person_agent import RAGResult


# ---------------------------------------------------------------------------
# Dataclass de evaluación
# ---------------------------------------------------------------------------

@dataclass
class EvalQuery:
    """Una query del set de evaluación."""
    question: str
    relevant_slugs: List[str] = field(default_factory=list)
    expected_answer: str = ""


# ---------------------------------------------------------------------------
# Métricas clásicas de IR (aplicadas a slugs)
# ---------------------------------------------------------------------------

def precision_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    """De los k recuperados, fracción que era relevante."""
    top_k = retrieved[:k]
    if not top_k:
        return 0.0
    rel = set(relevant)
    hits = sum(1 for s in top_k if s in rel)
    return hits / len(top_k)


def recall_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    """De los relevantes, fracción recuperada en los top-k."""
    if not relevant:
        return 0.0
    top_k = set(retrieved[:k])
    rel = set(relevant)
    return len(top_k & rel) / len(rel)


def reciprocal_rank(retrieved: List[str], relevant: List[str]) -> float:
    """1 / (posición del primer relevante)."""
    rel = set(relevant)
    for i, s in enumerate(retrieved, 1):
        if s in rel:
            return 1.0 / i
    return 0.0


def _dedupe_preserve_order(items: List[str]) -> List[str]:
    seen: set = set()
    out: List[str] = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


# ---------------------------------------------------------------------------
# Evaluación por query
# ---------------------------------------------------------------------------

def evaluate_query(eq: EvalQuery, k: int = 4) -> Dict:
    """
    Corre la query por el grafo y reporta:
      - routing: qué agentes seleccionó el router
      - retrieval: qué slugs aparecieron en los chunks agregados
      - métricas IR sobre la lista de slugs recuperados (por agente invocado)
    """
    # Ejecutamos el grafo completo
    output = rag_graph.invoke({
        "question": eq.question,
        "chat_history": [],
        "target_slugs": [],
        "route_type": "",
        "individual_results": [],
        "final_results": [],
    })
    results: List[RAGResult] = output["final_results"]
    individual: List[RAGResult] = output["individual_results"]

    # Slugs seleccionados por el router (orden del state)
    selected_slugs = [r.agent_slug for r in individual]

    # Slugs presentes en los chunks recuperados (desde metadata)
    retrieved_slugs: List[str] = []
    for r in individual:
        for doc in r.source_documents:
            retrieved_slugs.append(doc.metadata.get("agent", "?"))
    retrieved_slugs = _dedupe_preserve_order(retrieved_slugs)

    # Routing correcto si el set seleccionado == set relevante
    routing_correct = set(selected_slugs) == set(eq.relevant_slugs)

    return {
        "question": eq.question,
        "relevant_slugs": eq.relevant_slugs,
        "selected_slugs": selected_slugs,
        "retrieved_slugs": retrieved_slugs,
        "routing_correct": int(routing_correct),
        f"precision@{k}": precision_at_k(retrieved_slugs, eq.relevant_slugs, k),
        f"recall@{k}": recall_at_k(retrieved_slugs, eq.relevant_slugs, k),
        "mrr": reciprocal_rank(retrieved_slugs, eq.relevant_slugs),
        "final_answer": results[0].answer if results else "",
    }


def evaluate_all(
    eval_set: List[EvalQuery],
    k: int = 4,
    verbose: bool = True,
) -> pd.DataFrame:
    """Evalúa el set completo y devuelve un DataFrame."""
    rows = []
    for i, eq in enumerate(eval_set, 1):
        if verbose:
            print(f"[{i}/{len(eval_set)}] {eq.question[:70]}...")
        rows.append(evaluate_query(eq, k=k))
    return pd.DataFrame(rows)


def summary_metrics(df: pd.DataFrame, k: int = 4) -> Dict[str, float]:
    """Agrega las métricas sobre el set completo."""
    return {
        "routing_accuracy": float(df["routing_correct"].mean()),
        f"precision@{k}_mean": float(df[f"precision@{k}"].mean()),
        f"recall@{k}_mean": float(df[f"recall@{k}"].mean()),
        "mrr_mean": float(df["mrr"].mean()),
        "n_queries": len(df),
    }
