# common/__init__.py
"""
Common utilities package: exposes shared configuration, document processing,
embedding generation, and vector store interfaces.
"""
from .config import settings
from .processor import DocumentProcessor
from .embedding_generator import EmbeddingGenerator
from .vector_store import VectorStore

__all__ = [
    "settings",
    "DocumentProcessor",
    "EmbeddingGenerator",
    "VectorStore",
]
