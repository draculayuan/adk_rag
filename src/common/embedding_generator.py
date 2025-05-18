from typing import List, Dict, Any
from google.cloud import aiplatform
from vertexai.language_models import TextEmbeddingModel
from .config import settings


class EmbeddingGenerator:
    def __init__(self):
        # self.project = settings.GOOGLE_CLOUD_PROJECT
        # self.location = settings.VERTEX_AI_LOCATION
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

    def generate_embeddings(
        self, chunks: List[Dict[str, Any]], chunk_batch_size=20
    ) -> List[Dict[str, Any]]:
        """Generate embeddings for text chunks."""
        print("Creating embeddings...")

        # Extract texts from chunks
        texts = [chunk["text"] for chunk in chunks]

        # Generate embeddings
        try:
            embeddings = self.embedding_model.get_embeddings(texts)
        except:
            # process in batch if too many tokens
            print("Token length too large for embedding - breaking down by batch")
            embeddings = []
            for i in range(0, len(texts), chunk_batch_size):
                batch = texts[i : i + chunk_batch_size]  # slice is safe at end
                batch_emb = self.embedding_model.get_embeddings(
                    batch
                )  # returns list/array
                embeddings.extend(batch_emb)

        # Combine embeddings with original chunk data
        for chunk, embedding in zip(chunks, embeddings):
            chunk["embedding"] = embedding.values

        return chunks

    def generate_single_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        embedding = self.embedding_model.get_embeddings([text])[0]
        return embedding.values
