from retrieval.fusion import retrieve_hybrid

def retrieval_agent(state, llm, index):
    rewrite_prompt = f"""You are a search query optimizer.
Rewrite the following query to be more specific and better suited for document retrieval.
Return only the rewritten query, nothing else. No explanation, no punctuation at the end.
Original query: {state["original_query"]}
Rewritten query:"""

    rewritten = llm.complete(rewrite_prompt)
    rewritten_query = rewritten.text.strip()
    print(f"Rewritten query: {rewritten_query}")

    unique_chunks = retrieve_hybrid(index, rewritten_query, similarity_top_k=3)
    print(f"Retrieved {len(unique_chunks)} unique chunks")

    return {
        "original_query": state["original_query"],
        "query_type": state.get("query_type"),
        "should_block": state.get("should_block"),
        "block_reason": state.get("block_reason"),
        "rewritten_query": rewritten_query,
        "retrieved_chunks": unique_chunks,
        "draft_answer": state.get("draft_answer")
    }