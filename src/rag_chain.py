"""
rag_chain.py
------------
Ensambla el pipeline RAG usando LangChain Expression Language (LCEL).

Pipeline por query:
    question
        -> (opcional) reformulación con historial para queries de follow-up
        -> retriever        (top-k chunks desde Pinecone)
        -> prompt con contexto + fuentes
        -> ChatGroq         (generación)
        -> StrOutputParser  (texto plano)

Mejoras sobre un RAG naive:
  - History-aware retriever: reformula la pregunta usando el historial antes
    de hacer retrieval, para que queries como "¿y dónde estudió?" funcionen.
  - El contexto inyectado incluye el filename del CV de cada chunk, para que
    el LLM pueda citar las fuentes en sus respuestas.
"""

from dataclasses import dataclass
from typing import List

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_groq import ChatGroq

from src import config
from src.retriever import build_retriever

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

# Reformulación de query usando historial (para follow-ups).
_CONDENSE_SYSTEM = (
    "Dada una conversación previa y una nueva pregunta del usuario, "
    "reformulá la nueva pregunta como una pregunta independiente, "
    "que pueda entenderse sin el historial. "
    "NO respondas la pregunta, solo reformulala. "
    "Si la nueva pregunta ya es autocontenida, devolvela tal cual."
)

_CONDENSE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", _CONDENSE_SYSTEM),
    MessagesPlaceholder("chat_history"),
    ("human", "{question}"),
])

# Prompt principal de QA.
_SYSTEM_PROMPT = (
    "Sos un asistente experto en recursos humanos que responde preguntas sobre "
    "un conjunto de CVs.\n\n"
    "Reglas:\n"
    "1. Basate EXCLUSIVAMENTE en el contexto proporcionado de los CVs.\n"
    "2. Si la respuesta está en el contexto (aunque sea en un solo fragmento), "
    "RESPONDÉ con el dato o síntesis pertinente. NO evadas diciendo 'no tengo "
    "información' si el contenido está presente.\n"
    "3. Para preguntas GENERALES sobre roles, tareas o perfiles: SINTETIZÁ la "
    "información de los CVs que aplique. Ejemplo: si preguntan '¿qué hace un "
    "ingeniero en software?' y en los CVs hay tareas de un ingeniero de software, "
    "describí esas tareas citando a el/los candidato(s).\n"
    "4. Para preguntas ESPECÍFICAS (email, teléfono, GitHub, nombre): devolvé "
    "el dato exacto del contexto.\n"
    "5. Si la información realmente NO está en el contexto, respondé: "
    "'No encontré esa información en los CVs disponibles'.\n"
    "6. NO inventes datos que no estén en el contexto.\n"
    "7. Citá siempre las fuentes entre corchetes con el nombre del archivo, "
    "por ej.: [cv_juan_perez.pdf].\n"
    "8. Respondé en el mismo idioma de la pregunta, de forma concisa y precisa.\n\n"
    "EJEMPLOS DE COMPORTAMIENTO ESPERADO:\n\n"
    "Ejemplo 1 — Pregunta específica:\n"
    "Pregunta: '¿Cuál es el email de Valeria?'\n"
    "Contexto: '...Correo: v.dominguez.dev@outlook.com...'\n"
    "Respuesta correcta: 'El email de Valeria es v.dominguez.dev@outlook.com [cv3.pdf]'\n\n"
    "Ejemplo 2 — Pregunta general que requiere síntesis:\n"
    "Pregunta: '¿Qué hace un ingeniero en software?'\n"
    "Contexto: '...Senior Software Engineer en Mercado Libre: Desarrollo de APIs "
    "de alta concurrencia, liderazgo técnico en migración a microservicios, "
    "implementación de pipelines de CI/CD...'\n"
    "Respuesta correcta: 'Según los CVs disponibles, un ingeniero de software se "
    "dedica a tareas como desarrollo de APIs de alta concurrencia, liderazgo "
    "técnico en migraciones a microservicios, e implementación de pipelines "
    "CI/CD [cv3.pdf]. Por ejemplo, Valeria Domínguez trabaja en Mercado Libre "
    "en estas tareas desde 2022.'\n"
    "Respuesta INCORRECTA: 'No encontré esa información en los CVs disponibles'\n"
    "(es incorrecta porque la información SÍ está en el contexto, aunque la "
    "pregunta sea general; hay que sintetizar).\n\n"
    "Ejemplo 3 — Información realmente ausente:\n"
    "Pregunta: '¿Cuál es el salario actual de Valeria?'\n"
    "Contexto: (no menciona salarios)\n"
    "Respuesta correcta: 'No encontré esa información en los CVs disponibles.'\n\n"
    "Ejemplo 4 — Comparación entre candidatos:\n"
    "Pregunta: '¿Qué candidatos saben Python?'\n"
    "Contexto: '...[cv1] ... Python, Django ... [cv2] ... JavaScript, React ... "
    "[cv3] ... Python, FastAPI ...'\n"
    "Respuesta correcta: 'Los candidatos que saben Python son:\\n"
    "- [cv1.pdf]: maneja Python y Django\\n"
    "- [cv3.pdf]: maneja Python y FastAPI'\n"
)

_HUMAN_TEMPLATE = (
    "Contexto recuperado de los CVs:\n\n{context}\n\n"
    "Pregunta: {question}"
)

_QA_PROMPT = ChatPromptTemplate.from_messages([
    ("system", _SYSTEM_PROMPT),
    ("human", _HUMAN_TEMPLATE),
])


# ---------------------------------------------------------------------------
# Dataclass de salida
# ---------------------------------------------------------------------------

@dataclass
class RAGResult:
    """Contenedor para la respuesta del pipeline RAG."""
    answer: str
    source_documents: List[Document]
    reformulated_question: str = ""  # útil para debugging


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------

def _format_docs_with_sources(docs: List[Document]) -> str:
    """
    Formatea los chunks recuperados incluyendo el filename de cada uno.
    Esto le permite al LLM citar las fuentes correctamente.
    """
    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "desconocido")
        page = doc.metadata.get("page", "?")
        parts.append(
            f"[Fragmento {i} | fuente: {source} | página: {page}]\n{doc.page_content}"
        )
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Chain builder
# ---------------------------------------------------------------------------

def build_chain(top_k: int = None, search_type: str = "similarity"):
    """
    Construye la cadena LCEL completa.

    Returns
    -------
    tuple[Runnable, VectorStoreRetriever, ChatGroq]
        (chain, retriever, llm) — el chain devuelve un str con la respuesta;
        retriever y llm se devuelven por separado para uso en evaluación.
    """
    retriever = build_retriever(top_k=top_k, search_type=search_type)

    llm = ChatGroq(
        model=config.GROQ_MODEL,
        api_key=config.GROQ_API_KEY,
        temperature=config.GROQ_TEMPERATURE,
    )

    # Sub-chain 1: reformulación de la pregunta si hay historial
    condense_chain = _CONDENSE_PROMPT | llm | StrOutputParser()

    def _get_question(inputs: dict) -> str:
        """Reformula si hay historial; si no, devuelve la pregunta tal cual."""
        if inputs.get("chat_history"):
            return condense_chain.invoke(inputs)
        return inputs["question"]

    # Sub-chain 2: QA sobre el contexto recuperado
    qa_chain = (
        {
            "context": retriever | _format_docs_with_sources,
            "question": RunnablePassthrough(),
        }
        | _QA_PROMPT
        | llm
        | StrOutputParser()
    )

    # Chain completo: reformular → QA
    chain = RunnableLambda(_get_question) | qa_chain

    return chain, retriever, llm


# ---------------------------------------------------------------------------
# Invoke público
# ---------------------------------------------------------------------------

def invoke(
    question: str,
    chain,
    retriever,
    chat_history: List = None,
) -> RAGResult:
    """
    Corre el pipeline RAG para una sola pregunta.

    Parameters
    ----------
    question : str
        Pregunta del usuario.
    chain : Runnable
        Cadena LCEL pre-construida por `build_chain()`.
    retriever : VectorStoreRetriever
        Retriever pre-construido por `build_chain()`.
    chat_history : list, optional
        Lista de HumanMessage / AIMessage con el historial previo.

    Returns
    -------
    RAGResult
    """
    if chat_history is None:
        chat_history = []

    inputs = {"question": question, "chat_history": chat_history}

    answer = chain.invoke(inputs)

    # Obtenemos los source documents por separado con la pregunta
    # reformulada si corresponde (aproximamos usando la original,
    # ya que LCEL no expone el intermedio directamente).
    source_docs = retriever.invoke(question)

    return RAGResult(
        answer=answer,
        source_documents=source_docs,
        reformulated_question=question,
    )


def invoke_without_rag(question: str, llm: ChatGroq = None) -> str:
    """
    Baseline sin retrieval: el LLM responde solo con su conocimiento general.
    Se usa en la evaluación para medir el delta de utilidad del RAG.
    """
    if llm is None:
        llm = ChatGroq(
            model=config.GROQ_MODEL,
            api_key=config.GROQ_API_KEY,
            temperature=config.GROQ_TEMPERATURE,
        )
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Sos un asistente de recursos humanos. Respondé la pregunta del usuario."),
        ("human", "{question}"),
    ])
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"question": question})
