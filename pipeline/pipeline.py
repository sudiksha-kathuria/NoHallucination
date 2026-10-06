import os
import numpy as np
from typing import TypedDict, List, Optional
from functools import partial

from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from sklearn.metrics.pairwise import cosine_similarity

from agents.router_agent import router_agent
from agents.retrieval_agent import retrieval_agent
from agents.synthesis_agent import synthesis_agent
from guardrails.input_guard import input_guard_agent
from guardrails.output_guard import output_guard_agent
from ingestion.embedder import get_embedding_model

load_dotenv()

qdrant_client = QdrantClient(host="localhost", port=6333)
embedding_model = get_embedding_model()
llm = Groq(model="qwen/qwen3-8b-8192", api_key=os.getenv("GROQ_API_KEY"), max_tokens=800)


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


def judge_agent(state):
    answer = state.get("draft_answer", " ")
    query = state.get("rewritten_query") or state.get("original_query", " ")
    chunks = state.get("retrieved_chunks", [])

    def faithfulness(ans, cks):
        if not cks or not ans:
            return 0.0
        a_tokens = set(ans.lower().split())
        c_tokens = set(" ".join(cks).lower().split())
        return round(len(a_tokens & c_tokens) / len(a_tokens), 4)

    def relevancy(q, a):
        q_emb = np.array(embedding_model.get_text_embedding(q)).reshape(1, -1)
        a_emb = np.array(embedding_model.get_text_embedding(a)).reshape(1, -1)
        return round(float(cosine_similarity(q_emb, a_emb)[0][0]), 4)

    def context_util(ans, cks):
        if not cks:
            return 0.0
        a_tokens = set(ans.lower().split())
        used = sum(1 for c in cks if len(a_tokens & set(c.lower().split())) > 2)
        return round(used / len(cks), 4)

    scores = {
        "faithfulness": faithfulness(answer, chunks),
        "relevancy": relevancy(query, answer),
        "context_utilisation": context_util(answer, chunks),
    }
    overall = round(float(np.mean(list(scores.values()))), 4)
    return {**state, "judge_scores": scores, "judge_overall": overall}


def _should_block(state):
    return "blocked" if state.get("should_block") else "continue"


def build_graph(collection_name: str):
    vector_store = QdrantVectorStore(client=qdrant_client, collection_name=collection_name)
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store, embed_model=embedding_model)

    g = StateGraph(PipelineState)
    g.add_node("router", partial(router_agent, llm=llm))
    g.add_node("input_guard", input_guard_agent)
    g.add_node("retrieval", partial(retrieval_agent, llm=llm, index=index))
    g.add_node("synthesis", partial(synthesis_agent, llm=llm))
    g.add_node("output_guard", output_guard_agent)
    g.add_node("judge", judge_agent)

    g.add_edge(START, "router")
    g.add_conditional_edges("router", _should_block, {"blocked": END, "continue": "input_guard"})
    g.add_conditional_edges("input_guard", _should_block, {"blocked": END, "continue": "retrieval"})
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