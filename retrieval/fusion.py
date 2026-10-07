from llama_index.core.vector_stores.types import VectorStoreQueryMode
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters, FilterOperator


def retrieve_for_session(index, query: str, document_ids: list, similarity_top_k: int = 5):
    """Dense retrieval filtered to only the session's active documents."""
    filters = MetadataFilters(
        filters=[
            MetadataFilter(key="document_id", value=doc_id, operator=FilterOperator.EQ)
            for doc_id in document_ids
        ],
        condition="or",  # match any of the active docs
    )
    retriever = index.as_retriever(
        vector_store_query_mode=VectorStoreQueryMode.DEFAULT,
        similarity_top_k=similarity_top_k,
        filters=filters,
    )
    results = retriever.retrieve(query)
    seen_ids = set()
    unique_chunks = []
    for r in results:
        if r.node.node_id not in seen_ids:
            seen_ids.add(r.node.node_id)
            unique_chunks.append(r.node.text)
    return unique_chunks