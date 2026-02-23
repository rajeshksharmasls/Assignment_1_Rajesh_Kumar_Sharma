# RAG Classic — Basics RAG Chatbot

A production-style Retrieval-Augmented Generation (RAG) chatbot built with **Pinecone**, **LangChain**, and **FastAPI**.

It ingests PDF documents, chunks them with page-number tracking, stores them in a Pinecone serverless index with integrated embedding, and answers questions with inline citations and page references.

---

## Architecture

```
  ┌────────────────── INGESTION PIPELINE ──────────────────┐
  │                                                        │
  │  PDF ──▶ Page Extraction ──▶ Chunking ──▶ Pinecone     │
  │          (per-page text)    (512 chars,   (upsert with  │
  │                              64 overlap)  integrated    │
  │                                           embedding)    │
  └────────────────────────────────────────────────────────┘

  ┌────────────────── QUERY PIPELINE ──────────────────────┐
  │                                                        │
  │  Question ──▶ Retrieval ──▶ Reranker ──▶ Generation    │
  │               (Pinecone     (BGE-M3)     (LangChain    │
  │                search)                    ChatOpenAI)   │
  │                                           with [1][2]   │
  └────────────────────────────────────────────────────────┘
```

| Stage      | What it does                                        | Model / Service                        |
| ---------- | --------------------------------------------------- | -------------------------------------- |
| Ingestion  | Extracts text per page from PDF, splits into chunks | `pypdf`                                |
| Embedding  | Stores chunks with server-side embedding            | Pinecone `multilingual-e5-large`       |
| Retrieval  | Semantic vector search over stored chunks           | Pinecone integrated search             |
| Reranking  | Re-orders results by true relevance to the query    | Pinecone `bge-reranker-v2-m3`          |
| Generation | Produces an answer with inline citations            | LangChain `ChatOpenAI` (`gpt-4o-mini`) |

---

## Project Structure

```
rag-classic/
├── app/
│   ├── __init__.py        # Package marker
│   ├── config.py          # Settings & environment variables
│   ├── ingestion.py       # PDF/TXT loading, page extraction, chunking
│   ├── embedding.py       # Create Pinecone index & upsert records
│   ├── retrieval.py       # Semantic vector search
│   ├── reranker.py        # Rerank with bge-reranker-v2-m3
│   ├── generation.py      # LLM answer generation with citations
│   └── api.py             # FastAPI REST endpoints
├── docs/                  # Source documents
│   ├── Student-Handbook-2025-206.pdf
│   └── Syllabus-Geol-2020.pdf
├├── main.py                # CLI entry point (ingest / ask / serve)
├── pyproject.toml         # Dependencies
├── .env                   # API keys (not committed — see .gitignore)
├── .gitignore
└── README.md
```

---

## Setup

### 1. Clone & install

```bash
git clone https://github.com/rajeshksharmasls/Assignment_1_Rajesh_Kumar_Sharma.git
cd Assignment_1_Rajesh_Kumar_Sharma

# Install with uv (recommended)
uv pip install -e .
```

### 2. Add API keys

Create a `.env` file in the project root:

```env
PINECONE_API_KEY=my-pinecone-api-key
OPENAI_API_KEY=my-openai-api-key
```

### 3. Verify config

```bash
python code/config
```

```
=== Config Test ===
PINECONE_API_KEY : ✅ set
OPENAI_API_KEY   : ✅ set
Index name       : rag-classic
Embed model      : multilingual-e5-large
Rerank model     : bge-reranker-v2-m3
Chunk size       : 512, overlap: 64
LLM model        : gpt-4o-mini
✅ Config loaded successfully!
```

---

## Quick Start

```bash
# 1. Install
uv pip install -e .

# 2. Ingest documents
python main.py ingest docs/Student-Handbook-2025-2026.pdf
python main.py ingest docs/Syllabus-Geol-2020.pdf

# 3. Ask a question
python main.py ask "What is the Teaching Schedule of Year 1?"

# 4. Or start the API server
python main.py serve
```

## CLI Usage

### Ingest a document

```bash
python main.py ingest docs/Studeent-Handbook-2025-2026.pdf
python main.py ingest docs/Syllabus-Geol-2020.pdf
```

### Ask a question

```bash
# Default (with reranking + citations)
python main.py ask "What is the Teaching Schedule for Year 1?"

# Debug mode — shows retrieval vs reranked comparison
python main.py ask "What is the Teaching Schedule for Year 1?" --debug

# Skip reranking — raw vector search only
python main.py ask "What is the Teaching Schedule for Year 1?" --no-rerank
```

```bash
python code/config                              # Verify env vars
python code/ingestion docs/Apple_Q24.pdf        # Test PDF extraction & chunking
python code/embedding                           # Test upsert (dummy data)
python code/retrieval "What is the Teaching Schedule for Year 1" # Test vector search
python code/reranker "What is the Teaching Schedule for Year 1"  # Test reranking
python code/generation                           # Test LLM generation
```

---

## API Server

```bash
python main.py serve
```

Server runs at **http://localhost:8000** — Swagger docs at **http://localhost:8000/docs**.

### Endpoints

| Method | Endpoint    | Description                                     |
| ------ | ----------- | ----------------------------------------------- |
| GET    | `/health`   | Health check                                    |
| POST   | `/ingest`   | Ingest a document by file path                  |
| POST   | `/chat`     | Ask a question (full RAG pipeline + debug mode) |
| POST   | `/generate` | Retrieve → rerank → generate (clean response)   |
| POST   | `/search`   | Search only (no generation)                     |

## Key Features

- **LangChain Integration** — Uses `langchain-openai` `ChatOpenAI` for LLM generation (latest API: `llm.invoke()`)
- **Integrated Embedding** — Pinecone handles embedding server-side via `multilingual-e5-large`; no local embedding model needed
- **Page Number Tracking** — Each chunk carries its source page number(s) through the entire pipeline
- **Reranking** — `bge-reranker-v2-m3` reorders retrieval results for better relevance
- **Inline Citations** — LLM answers include `[1]`, `[2]` references with source file and page numbers
- **Debug Mode** — `--debug` flag shows retrieval vs reranked comparison and position changes
- **`/generate` Endpoint** — Clean API endpoint returning answer, sources, and pipeline info
- **Skip Duplicate Ingestion** — `rag_test.py` checks if documents are already indexed before re-ingesting
