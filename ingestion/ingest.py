import os
import tempfile
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from ingestion.loader import load_documents
from ingestion.chunker import chunk_document
from ingestion.embedder import get_embedding_model

qdrant_client = QdrantClient(host="localhost", port=6333)
embedding_model = get_embedding_model()

def ingest_file_for_user(file_bytes, file_name, file_type, collection_name, document_id):
    suffix = f".{file_type}"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    try:
        documents = load_documents(input_dir=os.path.dirname(tmp_path))
        for doc in documents:
            doc.metadata["document_id"] = document_id
            doc.metadata["file_name"] = file_name
        all_nodes = []
        for doc in documents:
            chunks = chunk_document(doc, embedding_model)
            for chunk in chunks:
                chunk.metadata["document_id"] = document_id
            all_nodes.extend(chunks)
        _ensure_collection(collection_name)
        vector_store = QdrantVectorStore(client=qdrant_client, collection_name=collection_name)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        VectorStoreIndex(
            nodes=all_nodes,
            embed_model=embedding_model,
            storage_context=storage_context,
            show_progress=False,
        )
        print(f"[Ingest] {file_name} → {len(all_nodes)} chunks → '{collection_name}'")
    finally:
        os.unlink(tmp_path)

def _ensure_collection(collection_name):
    existing = [c.name for c in qdrant_client.get_collections().collections]
    if collection_name not in existing:
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )