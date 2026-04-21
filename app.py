"""
app.py
------
Interfaz Streamlit para el chatbot RAG sobre CVs.

El retriever y el chain se inicializan una sola vez por sesión usando
st.session_state para no recargar el modelo de embeddings en cada query.

Run:
    streamlit run app.py
"""

from pathlib import Path

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from src import config
from src.rag_chain import build_chain, invoke


# ---------------------------------------------------------------------------
# Config de la página
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="RAG CV Chatbot",
    page_icon="📄",
    layout="wide",
)

st.title("📄 RAG CV Chatbot")
st.caption("TP2 — Retrieval-Augmented Generation sobre CVs | CEIA-FIUBA")


# ---------------------------------------------------------------------------
# Sidebar: configuración e información
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("⚙️ Configuración")

    st.subheader("Retrieval")
    top_k = st.slider("Top-k chunks", 1, 10, config.TOP_K)
    search_type = st.selectbox(
        "Tipo de búsqueda",
        options=["similarity", "mmr"],
        index=0,
        help="MMR prioriza diversidad entre los chunks recuperados.",
    )

    st.subheader("LLM")
    st.text(f"Modelo: {config.GROQ_MODEL}")
    st.text(f"Temperature: {config.GROQ_TEMPERATURE}")

    st.divider()

    st.subheader("📂 Corpus")
    cvs_dir = Path(config.CVS_DIR)
    n_pdfs = len(list(cvs_dir.glob("*.pdf"))) if cvs_dir.exists() else 0
    st.metric("PDFs en corpus", n_pdfs)
    st.text(f"Index: {config.PINECONE_INDEX_NAME}")
    st.text(f"Namespace: {config.PINECONE_NAMESPACE}")

    st.divider()

    if st.button("🔄 Reinicializar cadena", use_container_width=True):
        for k in ("chain", "retriever", "llm", "_chain_config"):
            st.session_state.pop(k, None)
        st.rerun()

    if st.button("🗑️ Limpiar chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ---------------------------------------------------------------------------
# Inicialización del chain (una sola vez o cuando cambia la config)
# ---------------------------------------------------------------------------

current_cfg = (top_k, search_type)
if "chain" not in st.session_state or st.session_state.get("_chain_config") != current_cfg:
    with st.spinner("Conectando a Pinecone y cargando modelo de embeddings..."):
        try:
            chain, retriever, llm = build_chain(top_k=top_k, search_type=search_type)
            st.session_state.chain = chain
            st.session_state.retriever = retriever
            st.session_state.llm = llm
            st.session_state._chain_config = current_cfg
        except Exception as exc:
            st.error(f"Error al inicializar el sistema: {exc}")
            st.info("Verificá que corriste la ingesta: `python -m src.ingestion --force`")
            st.stop()


# ---------------------------------------------------------------------------
# Historial de chat
# ---------------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []


def _render_sources(docs):
    """Renderiza los chunks recuperados en un expander."""
    if not docs:
        return
    with st.expander(f"🔎 Ver {len(docs)} chunks recuperados"):
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "?")
            page = doc.metadata.get("page", "?")
            st.markdown(f"**Fragmento {i}** — `{source}` (pág. {page})")
            st.text(doc.page_content)
            st.divider()


# Render histórico
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            _render_sources(msg.get("sources", []))


# ---------------------------------------------------------------------------
# Nueva pregunta
# ---------------------------------------------------------------------------

question = st.chat_input("Preguntá algo sobre los candidatos...")

if question:
    # Mensaje del usuario
    with st.chat_message("user"):
        st.markdown(question)
    st.session_state.messages.append({"role": "user", "content": question})

    # Respuesta del asistente
    with st.chat_message("assistant"):
        with st.spinner("Buscando en los CVs y generando respuesta..."):
            try:
                # Armamos el chat_history en formato LangChain
                chat_history = []
                for msg in st.session_state.messages[:-1]:  # excluye la pregunta actual
                    if msg["role"] == "user":
                        chat_history.append(HumanMessage(content=msg["content"]))
                    elif msg["role"] == "assistant":
                        chat_history.append(AIMessage(content=msg["content"]))

                result = invoke(
                    question=question,
                    chain=st.session_state.chain,
                    retriever=st.session_state.retriever,
                    chat_history=chat_history,
                )
                st.markdown(result.answer)
                _render_sources(result.source_documents)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result.answer,
                    "sources": result.source_documents,
                })
            except Exception as exc:
                error_msg = f"Error al generar la respuesta: {exc}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                    "sources": [],
                })
