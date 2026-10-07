from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

qdrant_client = QdrantClient(host="localhost", port=6333)


def delete_document_from_qdrant(collection_name: str, document_id: str):
    """Removes all chunks tagged with document_id from the collection."""
    qdrant_client.delete(
        collection_name=collection_name,
        points_selector=Filter(
            must=[
                FieldCondition(
                    key="document_id",
                    match=MatchValue(value=document_id),
                )
            ]
        ),
    )
    print(f"[Retriever] Deleted chunks for doc {document_id} from '{collection_name}'")