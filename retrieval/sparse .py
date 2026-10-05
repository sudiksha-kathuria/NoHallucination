from llama_index.core.vector_stores.types import VectorStoreQueryMode

def get_sparse_retriever(index, similarity_top_k: int = 3):
    return index.as_retriever(
        vector_store_query_mode=VectorStoreQueryMode.SPARSE,
        similarity_top_k=similarity_top_k
    )

def retrieve_sparse(index, query: str, similarity_top_k: int = 3):
    retriever = get_sparse_retriever(index, similarity_top_k)
    return retriever.retrieve(query)