def get_dense_retriever(index, similarity_top_k: int = 3):
    return index.as_retriever(similarity_top_k=similarity_top_k)

def retrieve_dense(index, query: str, similarity_top_k: int = 3):
    retriever = get_dense_retriever(index, similarity_top_k)
    return retriever.retrieve(query)