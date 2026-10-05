from llama_index.core.node_parser import SemanticSplitterNodeParser

def chunk_document(doc, embed_model, threshold=95, min_chunks=3):
    chunker = SemanticSplitterNodeParser(
        buffer_size=1,
        breakpoint_percentile_threshold=threshold,
        embed_model=embed_model
    )
    chunks = chunker.get_nodes_from_documents([doc])
    if len(chunks) < min_chunks:
        aggressive_chunker = SemanticSplitterNodeParser(
            buffer_size=1,
            breakpoint_percentile_threshold=70,
            embed_model=embed_model
        )
        chunks = aggressive_chunker.get_nodes_from_documents([doc])
    return chunks

def chunk_all_documents(documents, embed_model):
    all_nodes = []
    for doc in documents:
        doc_chunks = chunk_document(doc, embed_model)
        print(f"{doc.metadata.get('file_name')}: {len(doc_chunks)} chunks")
        all_nodes.extend(doc_chunks)
    print(f"Total chunks: {len(all_nodes)}")
    return all_nodes