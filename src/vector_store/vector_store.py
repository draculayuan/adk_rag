from typing import List, Dict, Any
from google.cloud import aiplatform
from ..config import settings

class VectorStore:
    def __init__(self):
        self.project = settings.GOOGLE_CLOUD_PROJECT
        self.location = settings.VERTEX_AI_LOCATION
        self.index_endpoint = settings.VECTOR_SEARCH_INDEX_ENDPOINT
        self.index_id = settings.VECTOR_SEARCH_INDEX_ID
        
        # Initialize Vertex AI
        aiplatform.init(
            project=self.project,
            location=self.location
        )
        
        # Initialize the vector search index
        self.index = aiplatform.MatchingEngineIndex(
            index_endpoint=self.index_endpoint,
            index_id=self.index_id
        )

    def upsert_vectors(self, vectors: List[Dict[str, Any]]) -> None:
        """Upsert vectors into the vector store."""
        # Prepare vectors for upsert
        vector_ids = [f"{v['metadata']['file_name']}_{v['metadata']['chunk_index']}" for v in vectors]
        embeddings = [v['embedding'] for v in vectors]
        metadata = [v['metadata'] for v in vectors]
        
        # Upsert vectors
        self.index.upsert(
            ids=vector_ids,
            vectors=embeddings,
            metadata=metadata
        )

    def search_vectors(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar vectors."""
        # Perform vector search
        results = self.index.search(
            vector=query_embedding,
            num_neighbors=top_k
        )
        
        # Format results
        formatted_results = []
        for result in results:
            formatted_results.append({
                "text": result.metadata.get("text", ""),
                "metadata": result.metadata,
                "distance": result.distance
            })
            
        return formatted_results

    def delete_vectors(self, vector_ids: List[str]) -> None:
        """Delete vectors from the vector store."""
        self.index.delete(vector_ids) 