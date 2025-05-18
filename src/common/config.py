from pydantic_settings import BaseSettings
from typing import Optional
import os
from dotenv import load_dotenv

load_dotenv()


class Settings(BaseSettings):
    # Google Cloud settings
    GOOGLE_CLOUD_PROJECT: str = os.getenv("GOOGLE_CLOUD_PROJECT", "")
    GOOGLE_APPLICATION_CREDENTIALS: str = os.getenv(
        "GOOGLE_APPLICATION_CREDENTIALS", ""
    )

    # Vertex AI settings
    VERTEX_AI_LOCATION: str = os.getenv("VERTEX_AI_LOCATION", "us-central1")
    VERTEX_AI_ENDPOINT: str = os.getenv("VERTEX_AI_ENDPOINT", "")

    # Vector Search settings
    VECTOR_SEARCH_INDEX_ENDPOINT: str = os.getenv("VECTOR_SEARCH_INDEX_ENDPOINT", "")
    VECTOR_SEARCH_INDEX_ID: str = os.getenv("VECTOR_SEARCH_INDEX_ID", "")
    INDEX_DISPLAY_NAME: str = os.getenv("INDEX_DISPLAY_NAME", "")
    ENDPOINT_DISPLAY_NAME: str = os.getenv("ENDPOINT_DISPLAY_NAME", "")
    ENDPOINT_ID: str = os.getenv("ENDPOINT_ID", "")
    EMBEDDING_DIM: int = 768

    # LLM settings
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gemini-pro")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-005")

    # Document processing settings
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    SUPPORTED_FILE_TYPES: list = [".txt", ".pdf", ".docx", ".md"]

    # Web interface settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # Storage settings
    DOCUMENT_STORAGE_BUCKET: str = os.getenv("DOCUMENT_STORAGE_BUCKET", "")

    class Config:
        env_file = ".env"


settings = Settings()
