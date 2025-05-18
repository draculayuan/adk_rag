# agent/tools/retrieve.py
# from typing import List
# from agent.common.vector_store import VectorStore
# from agent.common.embedding_generator import EmbeddingGenerator

# Initialize shared utilities
# _vs = VectorStore()
# _embedder = EmbeddingGenerator()


def retrieve_documents(query: str):
    """
    Vector-search tool: returns the top_k text snippets relevant to the question.

    Args:
        query: User query text.
    Returns:
        List of document text snippets.
    """
    from vertexai.language_models import TextEmbeddingModel
    from google.cloud.aiplatform.matching_engine import MatchingEngineIndexEndpoint
    from google.cloud import firestore
    import vertexai

    vertexai.init(
        project="yuan-449301",
        location="us-central1",
    )

    _embedder = TextEmbeddingModel.from_pretrained("text-embedding-005")
    _endpoint = MatchingEngineIndexEndpoint(index_endpoint_name="7694472531129925632")
    _db = firestore.Client()

    query_embedding = _embedder.get_embeddings([query])[0].values
    response = _endpoint.find_neighbors(
        deployed_index_id="deployed_index_1747401318896",
        queries=[query_embedding],
        num_neighbors=3,
    )

    retrieved_results = []
    for response_ in response[0]:
        r = response_.id
        r = _db.collection("rag").document(r).get().to_dict()
        retrieved_results.append(
            {
                "text": r["text"],
                "file_name": r["file_name"],
                "file_path": r["file_path"],
            }
        )
    return retrieved_results
