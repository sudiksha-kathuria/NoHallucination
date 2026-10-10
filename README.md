# NoHallucination
A domain-agnostic RAG reliability layer that ensures every AI-generated answer is verifiably grounded in its source before it reaches the user. If the system cannot prove an answer came from the source, it refuses to respond rather than hallucinate.

## The Problem
When an AI system retrieves information and generates an answer, it can hallucinate. It might retrieve the right documents but generate an answer that contradicts them, or sound confident while being completely wrong. In high-stakes contexts, this causes real harm.
Most RAG systems deliver answers without verifying them. NoHallucination does not.

## The Solution
A multi-agent pipeline that retrieves, generates, guards, and judges every response before delivery. If an answer fails verification, the system retries or refuses. It never silently hallucinates.

## Architecture
```text
User Query
    ↓
Router Agent (classifies intent, detects injections)
    ↓
Retrieval Agent (hybrid dense + sparse search)
    ↓
Synthesis Agent (generates grounded answer)
    ↓
Guardrail Layer (PII detection, toxicity check)
    ↓
LLM-as-a-Judge (groundedness, relevance, completeness)
    ↓
Verified Answer or Transparent Refusal
```

## Tech Stack
| Component | Technology |
|---|---|
| Document ingestion | LlamaIndex |
| Vector database | Qdrant Cloud |
| Embeddings | BAAI/bge-small-en-v1.5 (HuggingFace) |
| Hybrid search | Dense + BM25 sparse via Qdrant |
| Agent orchestration | LangGraph |
| LLM | Qwen/qwen3-8b via Groq |
| Input/Output guardrails | Microsoft Presidio, Detoxify |
| LLM evaluation | Code-based judge (ROUGE-1, cosine similarity) |
| API layer | FastAPI |
| Deployment | Railway |

## Live API
The backend is deployed and accessible at:

    https://nohallucination-production.up.railway.app/

## Key Features
- **Domain agnostic:** works with any documents in any format (PDF, DOCX, TXT, MD)
- **Smart chunking:** semantic chunking with automatic retry logic for short documents
- **Hybrid retrieval:** combines dense semantic search and sparse BM25 keyword search
- **Guardrail layer:** detects PII leaks, toxicity, and prompt injection attempts
- **LLM-as-a-Judge:** scores every answer for groundedness, relevance, and completeness
- **Retry loop:** automatically retries generation if judge score is below threshold
- **Transparent refusal:** returns source documents directly if answer cannot be verified
- **REST API:** fully deployed FastAPI backend with interactive docs at `/docs`

## Project Structure
```text
NoHallucination/
│
├── notebooks/
│   └── main.ipynb
│
├── ingestion/
│   ├── loader.py
│   ├── chunker.py
│   └── embedder.py
│
├── retrieval/
│   ├── dense.py
│   ├── sparse.py
│   └── fusion.py
│
├── agents/
│   ├── router.py
│   ├── retrieval_agent.py
│   └── synthesis_agent.py
│
├── guardrails/
│   ├── input_guard.py
│   └── output_guard.py
│
├── judge/
│   ├── evaluator.py
│   └── retry.py
│
├── api/
│   └── main.py
│
├── sample_docs/
├── test_docs/
├── requirements.txt
├── Procfile
├── railway.toml
├── nixpacks.toml
└── README.md
```

## Build Progress
- [x] Phase 1: Smart document ingestion and hybrid retrieval
- [x] Phase 2: Stateful multi-agent orchestration
- [x] Phase 3: Guardrail layer
- [x] Phase 4: LLM-as-a-Judge evaluator
- [x] Phase 5: FastAPI deployment on Railway with Qdrant Cloud
- [ ] Phase 6: React frontend

## Phase 1: What is Built

**Smart document ingestion**
- Format-agnostic loading via LlamaIndex SimpleDirectoryReader
- Semantic chunking that splits at natural topic boundaries
- Automatic retry logic: if a document produces fewer than 3 chunks at threshold 95, automatically retries at threshold 70

**Hybrid retrieval**
- Dense vector search using BAAI/bge-small-en-v1.5 embeddings
- Sparse BM25 keyword search via Qdrant FastEmbed
- Hybrid fusion combining both retrieval methods

## Phase 2: What is Built

**LangGraph multi-agent pipeline**
- Router Agent: classifies query as factual, conversational, or unclear; detects and blocks prompt injection attempts
- Retrieval Agent: rewrites the query for better retrieval, fetches top-3 hybrid chunks from Qdrant, deduplicates by node ID
- Synthesis Agent: generates a grounded answer using only retrieved context; strips model thinking tokens; refuses if context is insufficient

## Phase 3: What is Built

**Input guardrail — PII detection**
- Runs after the router and before retrieval
- Uses Microsoft Presidio's AnalyzerEngine with spaCy's en_core_web_lg English model
- Detects names, email addresses, phone numbers, locations, and other sensitive identifiers
- Blocks and returns entity types detected if PII is found

**Output guardrail — toxicity detection**
- Runs after synthesis and before the final response is returned
- Uses Detoxify's "original" model trained on the Jigsaw Toxic Comment Classification dataset
- Scores across six categories: toxicity, severe toxicity, obscene, threat, insult, and identity attack
- Blocks response if toxicity score exceeds 0.5

## Phase 4: What is Built

**Code-based evaluation (Judge Layer)**
- Runs after the output guardrail on every non-blocked response
- Faithfulness: fraction of answer tokens present in retrieved chunks (ROUGE-1 style precision)
- Relevancy: cosine similarity between query embedding and answer embedding
- Context Utilization: fraction of retrieved chunks with at least 3 overlapping tokens with the answer
- Overall score is the mean of all three, returned as judge_overall

## Phase 5: What is Built

**FastAPI backend**
- REST API wrapping the full LangGraph pipeline
- Endpoints for document ingestion and query answering
- Interactive API docs at `/docs`

**Deployment**
- Backend deployed on Railway via Nixpacks (no Docker required)
- Vector store migrated to Qdrant Cloud (free tier)
- Environment variables managed via Railway dashboard

## Problems Faced and How They Were Handled

**Groq model unavailable**
`llama-3.3-70b-versatile` was not available on the free tier. Switched to `qwen/qwen3-8b`, which produces `<think>...</think>` blocks in responses. Added stripping logic in the synthesis agent to clean these before returning.

**Qwen thinking tokens in rewritten queries**
The retrieval agent uses the LLM to rewrite queries before searching. Qwen sometimes wraps the rewritten query in `<think>` tags, degrading search quality. Fixed by stripping the think block from `rewritten_query` before retrieval.

**Qdrant hybrid search setup**
Setting up hybrid dense + sparse search required Qdrant FastEmbed as the sparse model. Initial setup failed because the vector store was not initialized with `enable_hybrid=True`. Fixed by passing the correct parameters to `QdrantVectorStore`.

**Semantic chunker producing too few chunks**
For short documents, `SemanticSplitterNodeParser` at threshold 95 produced fewer than 3 chunks. Added automatic retry at threshold 70.

**Duplicate chunks in Qdrant**
Re-running ingestion without clearing the collection caused duplicates. Fixed by deleting the collection and re-ingesting cleanly.

**LangGraph state not propagating between nodes**
Retrieved chunks showed as 0 in final state due to a typo — `retrieved_chunk` vs `retrieved_chunks`. LangGraph silently drops keys not in the TypedDict schema. Fixed by correcting the field name.

**Input_guard node never executed**
The conditional edge from the router still pointed directly to retrieval instead of input_guard. Fixed by correcting the router's "continue" branch.

**Judge receiving empty chunks**
A typo — `retrived_chunks` instead of `retrieved_chunks` — caused the judge to default to an empty list silently. Fixed by correcting the spelling.

**Docker on macOS M-series**
`sentence-transformers` pulls GPU/CUDA torch on Linux, causing multi-GB downloads and disk full errors in Docker. Fixed by pinning `torch --index-url https://download.pytorch.org/whl/cpu` before `sentence-transformers` in requirements.txt. Abandoned local Docker in favour of Railway deployment.

**Railway deployment — no start command detected**
Railpack could not detect the FastAPI app in the `api/` subfolder. Fixed by adding `nixpacks.toml` and `railway.toml` with an explicit start command, and removing `.dockerignore` which was causing Railway to switch to Docker build mode.

## Setup

**Prerequisites**
- Python 3.11+

**Installation**

Clone the repo:

    git clone https://github.com/sudiksha-kathuria/NoHallucination.git
    cd NoHallucination

Create virtual environment:

    python3.11 -m venv .venv
    source .venv/bin/activate

Install dependencies:

    pip install -r requirements.txt

**Environment variables**

Create a `.env` file in the project root:

    SUPABASE_URL=your_supabase_url
    SUPABASE_ANON_KEY=your_supabase_anon_key
    SUPABASE_SERVICE_KEY=your_supabase_service_key
    GROQ_API_KEY=your_groq_api_key
    QDRANT_URL=your_qdrant_cloud_url
    QDRANT_API_KEY=your_qdrant_api_key

**Adding your own documents**
Place your documents in the `test_docs/` folder. Supported formats: PDF, DOCX, TXT, MD. This folder is gitignored. Sample documents are in `sample_docs/`.

**Running locally**

    uvicorn api.main:app --reload

Then open `http://localhost:8000/docs`.

## Motivation
This project was built to solve a real problem: AI systems that sound confident while being wrong. A system that refuses to answer when it cannot verify its output is more valuable than one that always generates something. NoHallucination is built on that principle.

## Author
Sudiksha Kathuria