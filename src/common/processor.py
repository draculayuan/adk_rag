from typing import List, Dict, Any
import PyPDF2
from docx import Document
import markdown
import os
import pandas as pd
import pytesseract
from PIL import Image
from langchain.text_splitter import RecursiveCharacterTextSplitter
from .config import settings

class DocumentProcessor:
    def __init__(self):
        #self.supported_types = settings.SUPPORTED_FILE_TYPES
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=50,
            length_function=len,
            is_separator_regex=False,
        )

    def process_document(self, root_path: str) -> List[Dict[str, Any]]:
        """Process a document and return chunks with metadata."""
        print("Processing data...")
        if not os.path.exists(root_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        chunks_collection = []
        for root, dirs, files in os.walk(root_path):
            for file in files:
                file_path = os.path.join(root, file)
                file_ext = os.path.splitext(file_path)[1].lower()
                #if file_ext not in self.supported_types:
                    #raise ValueError(f"Unsupported file type: {file_ext}")

                # Extract text based on file type
                if file_ext == '.pdf':
                    text = self._extract_pdf_text(file_path)
                elif file_ext == '.docx':
                    text = self._extract_docx_text(file_path)
                elif file_ext == '.md':
                    text = self._extract_markdown_text(file_path)
                elif file_ext == '.csv':
                    text = self._extract_csv(file_path)
                elif file_ext in ['.png', 'jpeg', 'jpg']:
                    text = self._extract_image(file_path)
                else:  # .txt
                    text = self._extract_text_file(file_path)

                # Chunk the text
                chunks = self._chunk_text(text)
                chunks_collection.append(
                    {
                        "chunks": chunks,
                        "file_path": file_path
                    }
                )

        # Add metadata to chunks
        return self._add_metadata(chunks_collection)

    def _extract_csv(self, file_path: str) -> str:
        texts = ""
        df = pd.read_csv(file_path)
        for i, row in df.iterrows():
            text = " | ".join([f"{col}: {row[col]}" for col in df.columns])
            texts += '\n ' + (text)
        return texts
            
    def _extract_image(self, file_path: str) -> str:
        img = Image.open(file_path)
        text = pytesseract.image_to_string(img)
        return text
        
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
        chunks = self.text_splitter.split_text(text)

        return chunks

    def _add_metadata(self, chunks_collection: List[dict]) -> List[Dict[str, Any]]:
        """Add metadata to each chunk."""
        results = []
        idx = 0
        for data in chunks_collection:
            chunks = data["chunks"]
            file_path = data["file_path"]
            file_name = os.path.basename(file_path)
            for chunk in chunks:
                results.append(
                    {
                        "text": chunk,
                        "metadata": {
                            "source": file_path,
                            "file_name": file_name,
                            "chunk_index": idx
                        }
                    }
                )
                idx += 1
        return results