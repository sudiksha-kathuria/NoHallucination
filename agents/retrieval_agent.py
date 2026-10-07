from retrieval.fusion import retrieve_for_session

def retrieval_agent(state, llm, index):
    rewrite_prompt = f"""You are a search query optimizer.
Rewrite the following query to be more specific and better suited for document retrieval.
Return only the rewritten query, nothing else. No explanation, no punctuation at the end.
Original query: {state["original_query"]}
Rewritten query:"""

    rewritten = llm.complete(rewrite_prompt)
    rewritten_query = rewritten.text.strip()
    print(f"Rewritten query: {rewritten_query}")

    document_ids = state.get("document_ids") or []
    unique_chunks = retrieve_for_session(index, rewritten_query, document_ids, similarity_top_k=5)
    print(f"Retrieved {len(unique_chunks)} unique chunks")

    return {
        **state,
        "rewritten_query": rewritten_query,
        "retrieved_chunks": unique_chunks,
    }