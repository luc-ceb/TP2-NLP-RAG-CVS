"""
app.py
------
Interfaz Streamlit para el chatbot multi-agente RAG sobre CVs.

El orquestador se inicializa una sola vez por sesión y los agentes se
pre-construyen al arrancar para evitar latencia en la primera query.

Run:
    streamlit run app.py
"""

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from src import config
from src.agents.orchestrator import preload_agents, run
from src.agents.registry import get_all_slugs, get_display_name


# ---------------------------------------------------------------------------
# Página
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="RAG CV Chatbot — Multi-Agente",
    page_icon="📄",
    layout="wide",
)

st.title("📄 RAG CV Chatbot — Multi-Agente")
st.caption(
    "TP3 — Un agente por candidato · LangGraph + Pinecone + Groq/Llama 3.1 · CEIA-FIUBA"
)


# ---------------------------------------------------------------------------
# Sidebar: info de agentes registrados
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("🤖 Agentes registrados")
    for slug in get_all_slugs():
        st.markdown(f"• **{get_display_name(slug)}**  (`{slug}`)")
    st.divider()

    st.header("⚙️ Configuración")
    st.text(f"LLM: {config.GROQ_MODEL}")
    st.text(f"Index: {config.PINECONE_INDEX_NAME}")
    st.text(f"Top-K: {config.TOP_K}")
    st.divider()

    if st.button("🗑️ Limpiar chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ---------------------------------------------------------------------------
# Pre-carga de agentes (una vez por sesión)
# ---------------------------------------------------------------------------

if "agents_ready" not in st.session_state:
    with st.spinner("Cargando agentes y conectando a Pinecone..."):
        try:
            preload_agents()
            st.session_state.agents_ready = True
        except Exception as exc:
            st.error(f"Error al inicializar los agentes: {exc}")
            st.info(
                "Verificá que corriste la ingesta: "
                "`python -m src.ingestion --agent all --force`"
            )
            st.stop()


# ---------------------------------------------------------------------------
# Historial de chat
# ---------------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []


def _render_sources(docs, display_name: str):
    if not docs:
        return
    with st.expander(f"🔎 Contexto — {display_name}  ({len(docs)} chunks)"):
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "?")
            page = doc.metadata.get("page", "?")
            author = doc.metadata.get("author", "")
            label = f"**Fragmento {i}** — `{source}` (pág. {page})"
            if author:
                label += f"  *de {author}*"
            st.markdown(label)
            st.text(doc.page_content)
            st.divider()


# Render del histórico
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            for resp in msg["agent_responses"]:
                is_synthesis = resp.get("agent_slug") == "synthesis"
                if len(msg["agent_responses"]) > 1 or is_synthesis:
                    st.markdown(f"### {resp['display_name']}")
                st.markdown(resp["answer"])
                _render_sources(resp.get("sources", []), resp["display_name"])
        else:
            st.markdown(msg["content"])


# ---------------------------------------------------------------------------
# Nueva pregunta
# ---------------------------------------------------------------------------

question = st.chat_input("Preguntá sobre alguno de los candidatos...")

if question:
    # Mensaje del usuario
    with st.chat_message("user"):
        st.markdown(question)
    st.session_state.messages.append({"role": "user", "content": question})

    # Armamos el chat_history en formato LangChain (excluyendo la pregunta actual)
    chat_history = []
    for msg in st.session_state.messages[:-1]:
        if msg["role"] == "user":
            chat_history.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            combined = "\n\n".join(
                f"[{r['display_name']}]: {r['answer']}" for r in msg["agent_responses"]
            )
            chat_history.append(AIMessage(content=combined))

    # Invocamos el grafo
    with st.chat_message("assistant"):
        with st.spinner("Ruteando, recuperando contexto y generando respuesta..."):
            try:
                results = run(question=question, chat_history=chat_history)

                agent_responses = []
                for result in results:
                    is_synthesis = result.agent_slug == "synthesis"
                    if len(results) > 1 or is_synthesis:
                        st.markdown(f"### {result.display_name}")
                    st.markdown(result.answer)
                    _render_sources(result.source_documents, result.display_name)
                    agent_responses.append({
                        "agent_slug": result.agent_slug,
                        "display_name": result.display_name,
                        "answer": result.answer,
                        "sources": result.source_documents,
                    })

                st.session_state.messages.append({
                    "role": "assistant",
                    "agent_responses": agent_responses,
                })

            except Exception as exc:
                error_msg = f"Error al generar la respuesta: {exc}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "agent_responses": [{
                        "agent_slug": "error",
                        "display_name": "Sistema",
                        "answer": error_msg,
                        "sources": [],
                    }],
                })
