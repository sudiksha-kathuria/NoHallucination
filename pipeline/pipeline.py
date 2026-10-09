import os
from typing import TypedDict, List, Optional
from functools import partial

from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from llama_index.llms.groq import Groq

from ingestion.embedder import get_embedding_model
from agents.router_agent import router_agent
from agents.retrieval_agent import retrieval_agent
from agents.synthesis_agent import synthesis_agent
from guardrails.input_guard import input_guard_agent
from guardrails.output_guard import output_guard_agent
from judge.judge import judge_agent

load_dotenv()

qdrant_client = QdrantClient(host=os.getenv("QDRANT_HOST", "localhost"),port=int(os.getenv("QDRANT_PORT", 6333)))
embedding_model = get_embedding_model()
llm = Groq(model="llama3-8b-8192", api_key=os.getenv("GROQ_API_KEY"), max_tokens=800)


class PipelineState(TypedDict, total=False):
    original_query: str
    query_type: Optional[str]
    rewritten_query: Optional[str]
    retrieved_chunks: Optional[List[str]]
    draft_answer: Optional[str]
    should_block: Optional[bool]
    block_reason: Optional[str]
    pii_detected: Optional[bool]
    pii_entities: Optional[List[str]]
    is_toxic: Optional[bool]
    toxicity_scores: Optional[dict]
    judge_scores: Optional[dict]
    judge_overall: Optional[float]
    collection_name: Optional[str]
    document_ids: Optional[List[str]]


def _should_block(state):
    return "blocked" if state.get("should_block") else "continue"


def build_graph(collection_name: str):
    vector_store = QdrantVectorStore(client=qdrant_client, collection_name=collection_name)
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store, embed_model=embedding_model)

    g = StateGraph(PipelineState)
    g.add_node("router",       partial(router_agent, llm=llm))
    g.add_node("input_guard",  input_guard_agent)
    g.add_node("retrieval",    partial(retrieval_agent, llm=llm, index=index))
    g.add_node("synthesis",    partial(synthesis_agent, llm=llm))
    g.add_node("output_guard", output_guard_agent)
    g.add_node("judge",        partial(judge_agent, embedding_model=embedding_model))

    g.add_edge(START, "router")
    g.add_conditional_edges("router",       _should_block, {"blocked": END, "continue": "input_guard"})
    g.add_conditional_edges("input_guard",  _should_block, {"blocked": END, "continue": "retrieval"})
    g.add_edge("retrieval", "synthesis")
    g.add_edge("synthesis", "output_guard")
    g.add_conditional_edges("output_guard", _should_block, {"blocked": END, "continue": "judge"})
    g.add_edge("judge", END)

    return g.compile()


def run_pipeline(query: str, collection_name: str, document_ids: list) -> dict:
    graph = build_graph(collection_name)
    initial_state = {
        "original_query": query,
        "collection_name": collection_name,
        "document_ids": document_ids,
        "query_type": None,
        "rewritten_query": None,
        "retrieved_chunks": None,
        "draft_answer": None,
        "should_block": False,
        "block_reason": None,
        "pii_detected": None,
        "pii_entities": None,
        "is_toxic": None,
        "toxicity_scores": None,
        "judge_scores": None,
        "judge_overall": None,
    }
    return graph.invoke(initial_state)