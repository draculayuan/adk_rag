"""
Ingestion module for uploading data to VertexAI index.

This module provides functionalities to process and upload data 
to a VertexAI Vector Search index using the VectorStore.
"""
import os
import logging
from typing import List, Dict, Any, Optional, Tuple
import json
from pathlib import Path

from google.cloud import aiplatform

from ..common.vector_store import VectorStore
from ..common.config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class DataIngestion:
    """
    Handles the ingestion of data into the VertexAI vector index.
    """
    
    def __init__(self):
        """Initialize the DataIngestion with VectorStore."""
        self.vector_store = VectorStore()
        logger.info("Initialized DataIngestion with VectorStore")
    
    def ingest_from_json(self, json_file_path: str) -> int:
        """
        Ingest data from a JSON file.
        
        Args:
            json_file_path: Path to JSON file containing data to ingest
            
        Returns:
            Number of records ingested
        """
        try:
            with open(json_file_path, 'r') as f:
                data = json.load(f)
            
            if not isinstance(data, list):
                raise ValueError("JSON data must be a list of records")
            
            logger.info(f"Loaded {len(data)} records from {json_file_path}")
            return self.ingest_vectors(data)
        except Exception as e:
            logger.error(f"Error ingesting from JSON file {json_file_path}: {str(e)}")
            raise
    
    def ingest_vectors(self, vectors: List[Dict[str, Any]]) -> int:
        """
        Ingest vector data directly.
        
        Args:
            vectors: List of dictionaries containing 'embedding' and 'metadata'
            
        Returns:
            Number of records ingested
        """
        try:
            # Validate format
            for i, vector in enumerate(vectors):
                if "embedding" not in vector:
                    raise ValueError(f"Record at index {i} missing 'embedding' field")
                if "metadata" not in vector:
                    raise ValueError(f"Record at index {i} missing 'metadata' field")
                
                # Ensure embedding dimension matches the expected dimension
                if len(vector["embedding"]) != settings.EMBEDDING_DIM:
                    raise ValueError(
                        f"Embedding at index {i} has dimension {len(vector['embedding'])}, "
                        f"expected {settings.EMBEDDING_DIM}"
                    )
            
            # Perform the upsert
            self.vector_store.upsert_vectors(vectors)
            logger.info(f"Successfully ingested {len(vectors)} vectors")
            return len(vectors)
        except Exception as e:
            logger.error(f"Error ingesting vectors: {str(e)}")
            raise
    
    def delete_vectors(self, vector_ids: List[str]) -> int:
        """
        Delete vectors from the index.
        
        Args:
            vector_ids: List of vector IDs to delete
            
        Returns:
            Number of vectors deleted
        """
        try:
            self.vector_store.delete_vectors(vector_ids)
            logger.info(f"Successfully deleted {len(vector_ids)} vectors")
            return len(vector_ids)
        except Exception as e:
            logger.error(f"Error deleting vectors: {str(e)}")
            raise

def main():
    """Command line interface for data ingestion."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Ingest data into VertexAI Vector Search")
    parser.add_argument("--input", "-i", required=True, help="Path to JSON file with data to ingest")
    parser.add_argument("--delete", "-d", action="store_true", help="Delete vectors instead of ingesting")
    parser.add_argument("--id-file", help="Path to file containing vector IDs to delete (one per line)")
    
    args = parser.parse_args()
    
    ingestion = DataIngestion()
    
    if args.delete:
        if not args.id_file:
            logger.error("--id-file is required when using --delete")
            return 1
        
        with open(args.id_file, 'r') as f:
            vector_ids = [line.strip() for line in f if line.strip()]
        
        count = ingestion.delete_vectors(vector_ids)
        logger.info(f"Deleted {count} vectors")
    else:
        count = ingestion.ingest_from_json(args.input)
        logger.info(f"Ingested {count} vectors")
    
    return 0

if __name__ == "__main__":
    exit(main()) 