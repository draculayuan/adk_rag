# agent/tools/retrieve.py
from typing import List
#from google.adk.tools import function_tool
from common.vector_store import VectorStore
from common.embedding_generator import EmbeddingGenerator

# Initialize shared utilities
_vs = VectorStore()
_embedder = EmbeddingGenerator()

#@function_tool
def retrieve_documents(question: str, top_k: int = 5) -> List[str]:
    """
    Vector-search tool: returns the top_k text snippets relevant to the question.

    Args:
        question: User query text.
        top_k: Number of similar chunks to return.
    Returns:
        List of document text snippets.
    """
    query_emb = _embedder.generate_single_embedding(question)
    hits = _vs.search_vectors(query_emb, top_k=top_k)
    return [hit['metadata'].get('text', '') for hit in hits]