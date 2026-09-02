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
| LLM | llama-3.3-70b via Groq |
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
│   ├── phase1_exploration.ipynb
│   ├── phase2_exploration.ipynb
│   ├── phase3_exploration.ipynb
│   ├── phase4_exploration.ipynb
│   └── phase5_exploration.ipynb
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
- [ ] Phase 2: Stateful multi-agent orchestration
- [ ] Phase 3: Guardrail layer
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
