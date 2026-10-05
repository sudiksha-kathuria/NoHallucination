from llama_index.core.vector_stores.types import VectorStoreQueryMode

def get_hybrid_retriever(index, similarity_top_k: int = 3):
    return index.as_retriever(
        vector_store_query_mode=VectorStoreQueryMode.HYBRID,
        similarity_top_k=similarity_top_k
    )

def retrieve_hybrid(index, query: str, similarity_top_k: int = 3):
    retriever = get_hybrid_retriever(index, similarity_top_k)
    results = retriever.retrieve(query)
    seen_ids = set()
    unique_chunks = []
    for r in results:
        if r.node.node_id not in seen_ids:
            seen_ids.add(r.node.node_id)
            unique_chunks.append(r.node.text)
    return unique_chunks