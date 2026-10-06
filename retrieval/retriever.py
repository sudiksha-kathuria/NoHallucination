from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

qdrant_client = QdrantClient(host="localhost", port=6333)

def delete_document_from_qdrant(collection_name: str, document_id: str):
    qdrant_client.delete(
        collection_name=collection_name,
        points_selector=Filter(
            must=[
                FieldCondition(
                    key="document_id",
                    match=MatchValue(value=document_id)
                )
            ]
        )
    )