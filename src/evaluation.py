"""
evaluation.py
-------------
Evaluación del sistema RAG con métricas cuantitativas:
  - Retrieval: precision@k, recall@k, MRR
  - Generación: comparación cualitativa RAG vs no-RAG (baseline sin contexto)

La evaluación de retrieval requiere un ground truth definido manualmente:
para cada query, qué archivos de CV contienen la respuesta correcta.
Ese ground truth se arma en el notebook (sección 6).
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from src.rag_chain import invoke, invoke_without_rag


# ---------------------------------------------------------------------------
# Dataclass del set de evaluación
# ---------------------------------------------------------------------------

@dataclass
class EvalQuery:
    """Una query del set de evaluación."""
    question: str
    # Archivos de CV que contienen la respuesta correcta (ground truth)
    relevant_sources: List[str] = field(default_factory=list)
    # Respuesta esperada (opcional, solo para revisión cualitativa)
    expected_answer: str = ""


# ---------------------------------------------------------------------------
# Métricas de retrieval
# ---------------------------------------------------------------------------

def precision_at_k(retrieved_sources: List[str], relevant_sources: List[str], k: int) -> float:
    """
    Precision@k = (# relevantes en top-k) / k

    De los k documentos que recuperé, ¿qué fracción era realmente útil?
    """
    top_k = retrieved_sources[:k]
    if not top_k:
        return 0.0
    relevant_set = set(relevant_sources)
    hits = sum(1 for s in top_k if s in relevant_set)
    return hits / len(top_k)


def recall_at_k(retrieved_sources: List[str], relevant_sources: List[str], k: int) -> float:
    """
    Recall@k = (# relevantes en top-k) / (# relevantes totales)

    De todos los documentos relevantes que existían, ¿qué fracción logré recuperar?
    """
    if not relevant_sources:
        return 0.0
    top_k = set(retrieved_sources[:k])
    relevant_set = set(relevant_sources)
    hits = len(top_k & relevant_set)
    return hits / len(relevant_set)


def reciprocal_rank(retrieved_sources: List[str], relevant_sources: List[str]) -> float:
    """
    Reciprocal Rank = 1 / (posición del primer relevante)

    MRR (Mean Reciprocal Rank) = promedio de RR sobre todas las queries.
    Premia que la mejor respuesta aparezca arriba en el ranking.
    """
    relevant_set = set(relevant_sources)
    for i, s in enumerate(retrieved_sources, 1):
        if s in relevant_set:
            return 1.0 / i
    return 0.0


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------

def _dedupe_preserve_order(sources: List[str]) -> List[str]:
    """Elimina duplicados preservando el orden de aparición."""
    seen = set()
    out = []
    for s in sources:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


# ---------------------------------------------------------------------------
# Evaluación por query
# ---------------------------------------------------------------------------

def evaluate_query(
    eval_query: EvalQuery,
    chain,
    retriever,
    llm=None,
    k: int = 4,
    run_no_rag_baseline: bool = False,
) -> Dict:
    """Evalúa una sola query. Retorna un dict con métricas y respuestas."""
    result = invoke(eval_query.question, chain, retriever)

    # Sources únicas preservando el orden en que aparecen los chunks
    retrieved_sources = _dedupe_preserve_order(
        [Path(d.metadata.get("source", "")).name for d in result.source_documents]
    )

    metrics = {
        "question": eval_query.question,
        "relevant_sources": eval_query.relevant_sources,
        "retrieved_sources": retrieved_sources,
        f"precision@{k}": precision_at_k(retrieved_sources, eval_query.relevant_sources, k),
        f"recall@{k}": recall_at_k(retrieved_sources, eval_query.relevant_sources, k),
        "mrr": reciprocal_rank(retrieved_sources, eval_query.relevant_sources),
        "rag_answer": result.answer,
    }

    if run_no_rag_baseline:
        metrics["no_rag_answer"] = invoke_without_rag(eval_query.question, llm=llm)

    return metrics


# ---------------------------------------------------------------------------
# Evaluación sobre el set completo
# ---------------------------------------------------------------------------

def evaluate_all(
    eval_set: List[EvalQuery],
    chain,
    retriever,
    llm=None,
    k: int = 4,
    run_no_rag_baseline: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """Corre la evaluación sobre todo el set y devuelve un DataFrame."""
    rows = []
    for i, eq in enumerate(eval_set, 1):
        if verbose:
            print(f"[{i}/{len(eval_set)}] {eq.question[:70]}...")
        rows.append(evaluate_query(
            eq, chain, retriever, llm=llm, k=k,
            run_no_rag_baseline=run_no_rag_baseline,
        ))
    return pd.DataFrame(rows)


def summary_metrics(df: pd.DataFrame, k: int = 4) -> Dict[str, float]:
    """Agrega las métricas numéricas sobre el set completo."""
    return {
        f"precision@{k}_mean": float(df[f"precision@{k}"].mean()),
        f"recall@{k}_mean": float(df[f"recall@{k}"].mean()),
        "mrr_mean": float(df["mrr"].mean()),
        "n_queries": len(df),
    }
