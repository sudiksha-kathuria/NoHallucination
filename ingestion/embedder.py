from llama_index.embeddings.huggingface import HuggingFaceEmbedding

def get_embedding_model(model_name: str = "BAAI/bge-small-en-v1.5"):
    return HuggingFaceEmbedding(model_name=model_name)