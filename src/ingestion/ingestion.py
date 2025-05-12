"""
Ingestion module for uploading data to VertexAI index.

This module provides functionalities to process and upload data 
to a VertexAI Vector Search index using the VectorStore.
"""
import os
import logging
from typing import List, Dict, Any, Optional, Tuple, Union
import json
import csv
import pandas as pd
from pathlib import Path
import io
from PIL import Image
import base64

from google.cloud import aiplatform
try:
    from google.cloud import vision
except ImportError:
    vision = None

from common.vector_store import VectorStore
from common.config import settings
from common.processor import DocumentProcessor
from common.embedding_generator import EmbeddingGenerator

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
        self.document_processor = DocumentProcessor()
        self.embedding_generator = EmbeddingGenerator()
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
    
    def ingest_from_csv(self, csv_file_path: str, text_column: str = "text", metadata_columns: Optional[List[str]] = None) -> int:
        """
        Ingest data from a CSV file.
        
        Args:
            csv_file_path: Path to CSV file
            text_column: Column name containing the text to embed
            metadata_columns: List of column names to include as metadata
            
        Returns:
            Number of records ingested
        """
        try:
            # Read CSV file
            df = pd.read_csv(csv_file_path)
            logger.info(f"Loaded {len(df)} rows from {csv_file_path}")
            
            # Validate text column exists
            if text_column not in df.columns:
                raise ValueError(f"Text column '{text_column}' not found in CSV")
            
            # Prepare metadata columns
            if metadata_columns is None:
                metadata_columns = [col for col in df.columns if col != text_column]
            
            # Convert to the format expected by ingest_vectors
            vectors = []
            for idx, row in df.iterrows():
                text = row[text_column]
                if pd.isna(text) or text == "":
                    logger.warning(f"Skipping row {idx} due to empty text")
                    continue
                
                # Generate embedding for this text
                embedding = self.embedding_generator.generate_single_embedding(text)
                
                # Create metadata
                metadata = {
                    "text": text,
                    "source": csv_file_path,
                    "chunk_index": idx,
                    "file_name": os.path.basename(csv_file_path)
                }
                
                # Add additional metadata columns
                for col in metadata_columns:
                    if col in df.columns:
                        metadata[col] = str(row[col]) if not pd.isna(row[col]) else ""
                
                vectors.append({
                    "embedding": embedding,
                    "metadata": metadata
                })
            
            logger.info(f"Generated {len(vectors)} vectors from CSV data")
            return self.ingest_vectors(vectors)
        except Exception as e:
            logger.error(f"Error ingesting from CSV file {csv_file_path}: {str(e)}")
            raise
    
    def ingest_from_text_file(self, file_path: str) -> int:
        """
        Process and ingest a text file.
        
        Args:
            file_path: Path to the text file
            
        Returns:
            Number of records ingested
        """
        try:
            # Use the document processor to handle the text file
            chunks = self.document_processor.process_document(file_path)
            logger.info(f"Processed {len(chunks)} chunks from {file_path}")
            
            # Generate embeddings for the chunks
            chunks_with_embeddings = self.embedding_generator.generate_embeddings(chunks)
            
            # Convert to the format expected by ingest_vectors
            vectors = []
            for chunk in chunks_with_embeddings:
                vectors.append({
                    "embedding": chunk["embedding"],
                    "metadata": {
                        "text": chunk["text"],
                        "file_name": chunk["metadata"]["file_name"],
                        "chunk_index": chunk["metadata"]["chunk_index"],
                        "source": chunk["metadata"]["source"]
                    }
                })
            
            logger.info(f"Generated {len(vectors)} vectors from text file")
            return self.ingest_vectors(vectors)
        except Exception as e:
            logger.error(f"Error ingesting text file {file_path}: {str(e)}")
            raise
    
    def ingest_from_pdf(self, file_path: str) -> int:
        """
        Process and ingest a PDF file.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Number of records ingested
        """
        # PDF processing is the same as text file processing with our document processor
        return self.ingest_from_text_file(file_path)
    
    def ingest_from_image(self, image_path: str) -> int:
        """
        Process and ingest an image (PNG, JPEG, etc.).
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Number of records ingested
        """
        try:
            # 1. Extract text from image using Google Cloud Vision
            if vision is None:
                raise ImportError("Google Cloud Vision is required for image processing. Install with 'pip install google-cloud-vision'")
            
            client = vision.ImageAnnotatorClient()
            
            # Read the image file
            with open(image_path, "rb") as image_file:
                content = image_file.read()
            
            image = vision.Image(content=content)
            response = client.text_detection(image=image)
            
            if response.error.message:
                raise Exception(f"Error from Vision API: {response.error.message}")
            
            # Extract text
            extracted_text = ""
            for text in response.text_annotations:
                extracted_text = text.description
                break  # Only need the first one which contains all text
            
            if not extracted_text:
                logger.warning(f"No text detected in image: {image_path}")
                return 0
            
            logger.info(f"Extracted {len(extracted_text)} characters of text from image")
            
            # 2. Process the extracted text
            chunks = self._chunk_text(extracted_text, image_path)
            
            # 3. Generate embeddings
            vectors = []
            for i, chunk in enumerate(chunks):
                embedding = self.embedding_generator.generate_single_embedding(chunk)
                vectors.append({
                    "embedding": embedding,
                    "metadata": {
                        "text": chunk,
                        "file_name": os.path.basename(image_path),
                        "chunk_index": i,
                        "source": image_path,
                        "content_type": "image"
                    }
                })
            
            logger.info(f"Generated {len(vectors)} vectors from image")
            return self.ingest_vectors(vectors)
        except Exception as e:
            logger.error(f"Error ingesting image {image_path}: {str(e)}")
            raise
    
    def _chunk_text(self, text: str, source_path: str) -> List[str]:
        """Helper method to chunk text for image processing."""
        # Use document processor's chunking logic
        chunks = self.document_processor._chunk_text(text)
        return chunks
    
    def ingest_file(self, file_path: str, **kwargs) -> int:
        """
        Process and ingest a file based on its extension.
        
        Args:
            file_path: Path to the file
            **kwargs: Additional arguments passed to the specific ingestion method
            
        Returns:
            Number of records ingested
        """
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext == '.json':
            return self.ingest_from_json(file_path)
        elif file_ext == '.csv':
            return self.ingest_from_csv(file_path, **kwargs)
        elif file_ext in ['.txt', '.md']:
            return self.ingest_from_text_file(file_path)
        elif file_ext == '.pdf':
            return self.ingest_from_pdf(file_path)
        elif file_ext in ['.png', '.jpg', '.jpeg']:
            return self.ingest_from_image(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_ext}")
    
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
    parser.add_argument("--input", "-i", required=True, help="Path to file with data to ingest")
    parser.add_argument("--delete", "-d", action="store_true", help="Delete vectors instead of ingesting")
    parser.add_argument("--id-file", help="Path to file containing vector IDs to delete (one per line)")
    parser.add_argument("--text-column", default="text", help="Column name containing text for CSV files")
    parser.add_argument("--metadata-columns", help="Comma-separated list of column names to use as metadata for CSV files")
    
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
        kwargs = {}
        if args.metadata_columns:
            kwargs["metadata_columns"] = args.metadata_columns.split(",")
        if args.text_column:
            kwargs["text_column"] = args.text_column
            
        count = ingestion.ingest_file(args.input, **kwargs)
        logger.info(f"Ingested {count} vectors")
    
    return 0

if __name__ == "__main__":
    exit(main()) 