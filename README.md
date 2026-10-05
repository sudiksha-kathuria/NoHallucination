# NoHallucination
A domain-agnostic RAG reliability layer that ensures every AI-generated answer is verifiably grounded in its source before it reaches the user. If the system cannot prove an answer came from the source, it refuses to respond rather than hallucinate.

## The Problem
When an AI system retrieves information and generates an answer, it can hallucinate. It might retrieve the right documents but generate an answer that contradicts them, or sound confident while being completely wrong. In high-stakes contexts, this causes real harm.
Most RAG systems deliver answers without verifying them. NoHallucination does not.

## The Solution
A multi-agent pipeline that retrieves, generates, guards, and judges every response before delivery. If an answer fails verification, the system retries or refuses. It never silently hallucinates.

## Architecture
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

## Tech Stack
| Component | Technology |
|---|---|
| Document ingestion | LlamaIndex |
| Vector database | Qdrant |
| Embeddings | BAAI/bge-small-en-v1.5 (HuggingFace) |
| Hybrid search | Dense + BM25 sparse via Qdrant |
| Agent orchestration | LangGraph |
| LLM | Qwen/qwen3-8b-27b via Groq |
| Input/Output guardrails | Microsoft Presidio, Detoxify |
| LLM evaluation | LLM-as-a-Judge with RAGAS metrics |
| Observability | LangSmith |
| API layer | FastAPI |
| Containerisation | Docker |

## Key Features
- **Domain agnostic:** works with any documents in any format (PDF, DOCX, TXT, MD)
- **Smart chunking:** semantic chunking with automatic retry logic for short documents
- **Hybrid retrieval:** combines dense semantic search and sparse BM25 keyword search
- **Guardrail layer:** detects PII leaks, toxicity, and prompt injection attempts
- **LLM-as-a-Judge:** scores every answer for groundedness, relevance, and completeness
- **Retry loop:** automatically retries generation if judge score is below threshold
- **Transparent refusal:** returns source documents directly if answer cannot be verified
- **Full observability:** every decision traced and logged via LangSmith

## Project Structure

```text
NoHallucination/
│
├── notebooks/
│   ├── main.ipynb
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
├── qdrant_storage/
├── test_docs/
├── requirements.txt
├── docker-compose.yml
└── README.md
```

## Build Progress
- [x] Phase 1: Smart document ingestion and hybrid retrieval
- [x] Phase 2: Stateful multi-agent orchestration
- [x] Phase 3: Guardrail layer
- [ ] Phase 4: LLM-as-a-Judge evaluator
- [ ] Phase 5: Observability and deployment

## Phase 1: What is Built

**Smart document ingestion**
- Format-agnostic loading via LlamaIndex SimpleDirectoryReader
- Semantic chunking that splits at natural topic boundaries
- Automatic retry logic: if a document produces fewer than 3 chunks at threshold 95, automatically retries at threshold 70

**Hybrid retrieval**
- Dense vector search using BAAI/bge-small-en-v1.5 embeddings
- Sparse BM25 keyword search via Qdrant FastEmbed
- Hybrid fusion combining both retrieval methods
- Persistent storage via Docker volume mount

## Phase 2: What is Built

**LangGraph multi-agent pipeline**
- Router Agent: classifies query as factual, conversational, or unclear; detects and blocks prompt injection attempts
- Retrieval Agent: rewrites the query for better retrieval, fetches top-3 hybrid chunks from Qdrant, deduplicates by node ID
- Synthesis Agent: generates a grounded answer using only retrieved context; strips model thinking tokens; refuses if context is insufficient

**Graph structure**

START → router → (blocked → END) or (continue → retrieval) → synthesis → END

## Problems Faced and How They Were Handled

**Groq model unavailable**
`llama-3.3-70b-versatile` was not available on the free tier. Switched to `qwen/qwen3-8b-27b`, which works but produces `<think>...</think>` blocks in responses. Added stripping logic in the synthesis agent to clean these before returning the answer.

**Qwen thinking tokens in rewritten queries**
The retrieval agent uses the LLM to rewrite queries before searching. Qwen sometimes wraps the rewritten query in `<think>` tags, which then gets passed to the retriever and degrades search quality. Fixed by stripping the think block from `rewritten_query` before retrieval.

**Qdrant hybrid search setup**
Setting up hybrid dense + sparse search required Qdrant FastEmbed (`Qdrant/bm25`) as the sparse model. Initial setup failed because the vector store was not initialized with `enable_hybrid=True` and the sparse vector name was not configured. Fixed by passing the correct parameters to `QdrantVectorStore`.

**Semantic chunker producing too few chunks**
For short documents, `SemanticSplitterNodeParser` at threshold 95 produced fewer than 3 chunks, making retrieval ineffective. Added automatic retry logic: if a document produces fewer than 3 chunks at threshold 95, it retries at threshold 70.

**Duplicate chunks in Qdrant**
Re-running the ingestion cell without clearing the collection first caused the same documents to be stored multiple times under different node IDs. Deduplication by node ID in the retrieval agent only partially helped since all 3 retrieved results were duplicates of the same chunk. Fixed by deleting the collection and re-ingesting once cleanly.

**LangGraph state not propagating between nodes**
Retrieved chunks were showing as 0 in the final result even though the retrieval agent was finding 3 chunks. Root cause: a typo in `PipelineState` — the field was defined as `retrieved_chunk` (no s) but all agents used `retrieved_chunks`. LangGraph silently drops keys that are not in the TypedDict schema, so the chunks were never written to state. Fixed by correcting the field name.

**Docker container not persisting between sessions**
Qdrant data was lost between sessions because the container was started without a volume mount. Fixed by adding `-v $(pwd)/qdrant_storage:/qdrant/storage` and `--restart unless-stopped` to the docker run command.

## Setup

**Prerequisites**
- Python 3.11+
- Docker Desktop

**Installation**

Clone the repo:

    git clone https://github.com/sudiksha-kathuria/NoHallucination.git
    cd NoHallucination

Create virtual environment:

    python3.11 -m venv .venv
    source .venv/bin/activate

Install dependencies:

    pip install -r requirements.txt

Start Qdrant:

    docker run -p 6333:6333 \
      -v $(pwd)/qdrant_storage:/qdrant/storage \
      --restart unless-stopped \
      qdrant/qdrant

**Adding your own documents**
Place your documents in the `test_docs/` folder. Supported formats: PDF, DOCX, TXT, MD. This folder is gitignored to protect your privacy. Sample documents are provided in `sample_docs/`.

**Running the notebook**
Open `notebooks/phase1_exploration.ipynb` in VS Code, select the `.venv` kernel, and run cells in order.

## Motivation
This project was built to solve a real problem: AI systems that sound confident while being wrong. A system that refuses to answer when it cannot verify its output is more valuable than one that always generates something. NoHallucination is built on that principle.

## Author
Sudiksha Kathuria
