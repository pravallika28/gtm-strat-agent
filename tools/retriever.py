import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.anthropic import Anthropic

def query_gtm_docs(query_str: str):
    # Load docs from your local Mac folder
    documents = SimpleDirectoryReader("./data").load_data()
    index = VectorStoreIndex.from_documents(documents)
    
    query_engine = index.as_query_engine(
        llm=Anthropic(model="claude-3-5-sonnet-latest")
    )
    return str(query_engine.query(query_str))