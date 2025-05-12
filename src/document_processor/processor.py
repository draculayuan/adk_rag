from typing import List, Dict, Any
import PyPDF2
from docx import Document
import markdown
import os
from ..config import settings

class DocumentProcessor:
    def __init__(self):
        self.supported_types = settings.SUPPORTED_FILE_TYPES
        self.chunk_size = settings.CHUNK_SIZE
        self.chunk_overlap = settings.CHUNK_OVERLAP

    def process_document(self, file_path: str) -> List[Dict[str, Any]]:
        """Process a document and return chunks with metadata."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        file_ext = os.path.splitext(file_path)[1].lower()
        if file_ext not in self.supported_types:
            raise ValueError(f"Unsupported file type: {file_ext}")

        # Extract text based on file type
        if file_ext == '.pdf':
            text = self._extract_pdf_text(file_path)
        elif file_ext == '.docx':
            text = self._extract_docx_text(file_path)
        elif file_ext == '.md':
            text = self._extract_markdown_text(file_path)
        else:  # .txt
            text = self._extract_text_file(file_path)

        # Chunk the text
        chunks = self._chunk_text(text)
        
        # Add metadata to chunks
        return self._add_metadata(chunks, file_path)

    def _extract_pdf_text(self, file_path: str) -> str:
        """Extract text from PDF file."""
        text = ""
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        return text

    def _extract_docx_text(self, file_path: str) -> str:
        """Extract text from DOCX file."""
        doc = Document(file_path)
        return "\n".join([paragraph.text for paragraph in doc.paragraphs])

    def _extract_markdown_text(self, file_path: str) -> str:
        """Extract text from Markdown file."""
        with open(file_path, 'r', encoding='utf-8') as file:
            md_text = file.read()
            return markdown.markdown(md_text)

    def _extract_text_file(self, file_path: str) -> str:
        """Extract text from plain text file."""
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()

    def _chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks."""
        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length:
            end = start + self.chunk_size
            if end > text_length:
                end = text_length
            
            chunk = text[start:end]
            chunks.append(chunk)
            
            start = end - self.chunk_overlap

        return chunks

    def _add_metadata(self, chunks: List[str], file_path: str) -> List[Dict[str, Any]]:
        """Add metadata to each chunk."""
        file_name = os.path.basename(file_path)
        return [
            {
                "text": chunk,
                "metadata": {
                    "source": file_path,
                    "file_name": file_name,
                    "chunk_index": i,
                    "total_chunks": len(chunks)
                }
            }
            for i, chunk in enumerate(chunks)
        ] 