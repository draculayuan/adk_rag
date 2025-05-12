from typing import List, Dict, Any
from google.cloud import aiplatform
from vertexai.language_models import TextEmbeddingModel
from .config import settings

class EmbeddingGenerator:
    def __init__(self):
        #self.project = settings.GOOGLE_CLOUD_PROJECT
        #self.location = settings.VERTEX_AI_LOCATION
        self.model = settings.EMBEDDING_MODEL
        
        """
        # Initialize Vertex AI
        aiplatform.init(
            project=self.project,
            location=self.location
        )
        """
        
        # Initialize the embedding model
        self.embedding_model = TextEmbeddingModel.from_pretrained(self.model)

    def generate_embeddings(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate embeddings for text chunks."""
        # Extract texts from chunks
        texts = [chunk["text"] for chunk in chunks]
        
        # Generate embeddings
        embeddings = self.embedding_model.get_embeddings(texts)
        
        # Combine embeddings with original chunk data
        for chunk, embedding in zip(chunks, embeddings):
            chunk["embedding"] = embedding.values
            
        return chunks

    def generate_single_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        embedding = self.embedding_model.get_embeddings([text])[0]
        return embedding.values 