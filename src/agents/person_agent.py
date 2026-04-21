"""
person_agent.py
---------------
PersonAgent encapsula el pipeline RAG completo para UNA persona.

Cada agente:
    - lee de su propio namespace de Pinecone ("cv_{slug}")
    - usa un system prompt personalizado con el nombre de la persona
    - habla con Groq/Llama-3.1 vía ChatGroq
    - cachea retriever y chain tras el primer build()

Regla clave: cuando la query menciona a múltiples personas, CADA PersonAgent
debe responder SOLO sobre sí mismo. La comparación la hace synthesize_node.

Uso:
    agent = PersonAgent("cv1").build()
    result = agent.invoke("¿Qué experiencia tiene?", chat_history=[])
"""

from dataclasses import dataclass, field
from typing import List, Optional

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_groq import ChatGroq

from src import config
from src.agents.registry import get_display_name
from src.retriever import build_retriever


@dataclass
class RAGResult:
    """Contenedor para la respuesta de un agente."""
    agent_slug: str
    display_name: str
    answer: str
    source_documents: List[Document] = field(default_factory=list)


def _build_prompt(display_name: str) -> ChatPromptTemplate:
    system = (
        f"Sos un asistente especializado en el CV de {display_name}. "
        f"Tu única tarea es contar lo que el CV de {display_name} dice sobre él/ella.\n\n"
        "Reglas:\n"
        f"1. Si la pregunta pide comparar a {display_name} con otras personas, ignorá la parte "
        f"comparativa y respondé SOLO lo que corresponde a {display_name}. La comparación la "
        "hace otro sistema después; vos aportá los datos de tu persona.\n"
        f"2. Si la pregunta menciona a otras personas, ignoralas completamente y respondé "
        f"sobre {display_name}.\n"
        "3. Usá EXCLUSIVAMENTE la información del contexto. Si un dato concreto no está "
        "en el contexto, decí 'no figura en el CV'. No inventes.\n"
        f"4. Empezá tu respuesta con el nombre: '{display_name}: ...' y citá la fuente "
        "entre corchetes, ej. [cv2.pdf]."
    )
    human = (
        "Contexto del CV de {name}:\n\n{context}\n\n"
        "Pregunta del usuario: {question}\n\n"
        "Respondé describiendo lo que sabés sobre {name} según el contexto."
    )
    return ChatPromptTemplate.from_messages([
        ("system", system),
        MessagesPlaceholder("chat_history"),
        ("human", human),
    ])


def _format_docs_with_sources(docs: List[Document]) -> str:
    """
    Formatea los chunks recuperados incluyendo filename y página para que
    el LLM pueda citar las fuentes correctamente.
    """
    parts = []
    seen: set = set()
    for i, doc in enumerate(docs, 1):
        key = doc.page_content.strip()
        if key in seen:
            continue
        seen.add(key)
        source = doc.metadata.get("source", "desconocido")
        page = doc.metadata.get("page", "?")
        parts.append(
            f"[Fragmento {i} | fuente: {source} | página: {page}]\n{doc.page_content}"
        )
    return "\n\n".join(parts)


class PersonAgent:
    """
    Agente RAG para el CV de una sola persona.

    Parameters
    ----------
    slug : str
        Identificador del agente (clave de config.AGENTS).
    """

    def __init__(self, slug: str) -> None:
        if slug not in config.AGENTS:
            raise ValueError(f"Agente desconocido: '{slug}'")
        self.slug: str = slug
        self.display_name: str = get_display_name(slug)
        self._retriever: Optional[VectorStoreRetriever] = None
        self._chain = None

    def build(self) -> "PersonAgent":
        """
        Inicializa retriever y cadena LCEL. Idempotente — seguro llamar múltiples
        veces. Debe invocarse antes de `invoke()`.
        """
        if self._chain is not None:
            return self

        self._retriever = build_retriever(self.slug)

        llm = ChatGroq(
            model=config.GROQ_MODEL,
            api_key=config.GROQ_API_KEY,
            temperature=config.GROQ_TEMPERATURE,
        )
        prompt = _build_prompt(self.display_name)

        # LCEL chain: routea la question al retriever y al prompt en paralelo.
        self._chain = (
            {
                "context": (lambda x: x["question"]) | self._retriever | _format_docs_with_sources,
                "question": lambda x: x["question"],
                "chat_history": lambda x: x["chat_history"],
                "name": lambda x: self.display_name,
            }
            | prompt
            | llm
            | StrOutputParser()
        )
        return self

    def invoke(
        self,
        question: str,
        chat_history: Optional[List[BaseMessage]] = None,
    ) -> RAGResult:
        """
        Ejecuta el pipeline RAG para este agente sobre una pregunta puntual.

        Returns
        -------
        RAGResult
        """
        if self._chain is None:
            raise RuntimeError("Llamá a build() antes de invoke().")
        if chat_history is None:
            chat_history = []

        answer = self._chain.invoke({
            "question": question,
            "chat_history": chat_history,
        })

        # Recuperamos los source docs por separado para que la UI los muestre.
        raw_docs = self._retriever.invoke(question)
        seen: set = set()
        source_docs: List[Document] = []
        for doc in raw_docs:
            key = doc.page_content.strip()
            if key not in seen:
                seen.add(key)
                source_docs.append(doc)

        return RAGResult(
            agent_slug=self.slug,
            display_name=self.display_name,
            answer=answer,
            source_documents=source_docs,
        )
