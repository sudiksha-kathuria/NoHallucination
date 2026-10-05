from llama_index.core import SimpleDirectoryReader

def load_documents(input_dir: str):
    documents = SimpleDirectoryReader(input_dir=input_dir).load_data()
    return documents