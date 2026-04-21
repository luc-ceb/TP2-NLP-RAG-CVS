# TP2 — RAG sobre CVs

Chatbot basado en **Retrieval-Augmented Generation** sobre un corpus de currículums en PDF.

**Stack:** Pinecone (vector store) · Groq/Llama 3.1 (LLM) · HuggingFace MiniLM (embeddings) · LangChain (orquestación) · Streamlit (UI)

CEIA — FIUBA.
Autor: Ing. Luciano Ceballos

## Arquitectura

```
┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────┐
│  data/cvs/   │ → │ PyPDFLoader  │ → │  Splitter    │ → │  MiniLM  │
│    *.pdf     │   │ + limpieza   │   │ (500/50 ó    │   │ 384 dim  │
│              │   │              │   │  semántico)  │   │          │
└──────────────┘   └──────────────┘   └──────────────┘   └─────┬────┘
                                                                 │
                                                      ┌──────────▼───────┐
                                                      │    Pinecone      │
                                                      │  (serverless)    │
                                                      └──────────┬───────┘
                                                                 │
┌──────────┐   ┌──────────────┐   ┌──────────────┐     ┌─────────▼────┐
│  query   │ → │ (reformular  │ → │  retrieve    │ ←── │   index      │
│          │   │ si follow-up)│   │   top-k      │     │              │
└──────────┘   └──────────────┘   └───────┬──────┘     └──────────────┘
                                          │
                                 ┌────────▼──────────┐
                                 │  Groq Llama 3.1   │
                                 │  + system prompt  │
                                 └────────┬──────────┘
                                          │
                                 ┌────────▼──────────┐
                                 │ respuesta citada  │
                                 └───────────────────┘
```

## Estructura

```
rag-cvs/
├── data/
│   └── cvs/                  # cargar aqui los documentos en formato PDFs
├── notebooks/
│   └── tp2_rag_cvs.ipynb     # notebook principal (desarrollo + evaluación)
├── src/
│   ├── __init__.py
│   ├── config.py             # variables de entorno y parámetros
│   ├── ingestion.py          # carga → limpieza → chunk → embed → Pinecone
│   ├── retriever.py          # wrapper sobre el índice de Pinecone
│   ├── rag_chain.py          # cadena LCEL con history-aware retriever
│   ├── semantic_chunker.py   # chunker semántico (MiniLM, opcional)
│   └── evaluation.py         # métricas precision@k, recall@k, MRR
├── app.py                    # app Streamlit
├── requirements.txt
├── .env.example
└── README.md
```

## Instalación

```bash
# 1. Virtualenv
python -m venv .venv
source .venv/bin/activate

# 2. Dependencias
pip install -r requirements.txt

# 3. API keys
cp .env.example .env
#   GROQ_API_KEY      
#   PINECONE_API_KEY  
```

## Uso

### 1. Carga CVs

Copia los PDFs a `data/cvs/`:

```
data/cvs/
├── cv1.pdf
├── cv2.pdf
└── cv3.pdf
```

### 2. Ingesta a Pinecone

```bash
# Chunking clasico por caracteres
python -m src.ingestion --force

# Chunking semántico (mismo modelo de embeddings que el retriever)
python -m src.ingestion --force --use-semantic

# Directorio custom
python -m src.ingestion --cvs-dir /otra/ruta --force
```

El flag `--force` borra los vectores previos del namespace para evitar duplicados en re-ingestas.

### 3. Notebook (desarrollo + evaluación)

```bash
jupyter notebook notebooks/tp2_rag_cvs.ipynb
```

El notebook:
1. Arma y explica la arquitectura.
2. Ingresa los CVs a Pinecone.
3. Inspecciona manualmente el retriever.
4. Prueba el pipeline end-to-end.
5. Valida conversación con follow-ups (history-aware).
6. **Evalúa con métricas** (precision@k, recall@k, MRR) + comparación RAG vs no-RAG.

### 4. App Streamlit

```bash
streamlit run app.py
```

Permite:
- Configurar top-k y tipo de búsqueda (similarity / MMR) en vivo.
- Chat conversacional con follow-ups.
- Ver los chunks recuperados con sus fuentes y páginas.

## Uso programático

```python
from src.ingestion import ingest
from src.rag_chain import build_chain, invoke

# Ingesta (una sola vez, o con cambios)
ingest(cvs_dir="data/cvs", force=True)

# Construir la cadena
chain, retriever, llm = build_chain()

# Query
result = invoke("¿Qué candidatos saben Python?", chain, retriever)
print(result.answer)
print([d.metadata['source'] for d in result.source_documents])
```

## Evaluación

El módulo `src/evaluation.py` implementa tres métricas clásicas de information retrieval:

| Métrica | Fórmula | Qué mide |
|---------|---------|----------|
| **Precision@k** | # relevantes en top-k / k | Señal / ruido en los chunks recuperados |
| **Recall@k** | # relevantes en top-k / # relevantes totales | Cobertura de la recuperación |
| **MRR** | promedio de 1/(rango del 1er relevante) | Qué tan arriba aparece la mejor respuesta |

Para medirlas se arma un `eval_set` manual con queries y sus *ground truth sources* (qué archivos contienen la respuesta). Ver sección 6 del notebook.

## Decisiones de diseño y tradeoffs

- **Embeddings locales** vs API: `paraphrase-multilingual-MiniLM-L12-v2` corre gratis y sirve para español. Si el corpus crece a miles de CVs conviene probar `BAAI/bge-m3` o embeddings de OpenAI.
- **Pinecone** vs Chroma: Pinecone serverless es managed, persistente y escalable sin infra; Chroma es más simple para local. Acá elegimos Pinecone por alineación con el enunciado/contenido visto en clases.
- **Chunk 500 / overlap 50**: secciones de CV son cortas y densas. Chunks grandes diluyen la señal; chunks chicos fragmentan contexto. 500/50 es un punto razonable en la literatura.
- **Llama 3.1 8B via Groq**: tier gratuito, latencia < 1s. Para calidad máxima conviene `llama-3.3-70b-versatile`.
- **System prompt estricto**: obliga a citar fuentes entre corchetes `[filename.pdf]` y a rehusarse si la info no está en el contexto, mitigando alucinaciones.
- **History-aware retriever**: el retriever naive falla en follow-ups ("¿y dónde estudió?"). La reformulación vía LLM antes del retrieve resuelve esto.