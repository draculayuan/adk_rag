from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
import uvicorn
import os
from typing import List, Dict, Any

from config import settings
from document_processor.processor import DocumentProcessor
from embedding.embedding_generator import EmbeddingGenerator
from vector_store.vector_store import VectorStore
from llm.llm_engine import LLMEngine

app = FastAPI(title="Cymbal Knowledge Bot")

# Initialize components
document_processor = DocumentProcessor()
embedding_generator = EmbeddingGenerator()
vector_store = VectorStore()
llm_engine = LLMEngine()

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Render the home page."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload and process a document."""
    try:
        # Save the uploaded file temporarily
        file_path = f"temp/{file.filename}"
        os.makedirs("temp", exist_ok=True)
        
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Process the document
        chunks = document_processor.process_document(file_path)
        
        # Generate embeddings
        chunks_with_embeddings = embedding_generator.generate_embeddings(chunks)
        
        # Store in vector database
        vector_store.upsert_vectors(chunks_with_embeddings)
        
        # Clean up
        os.remove(file_path)
        
        return {"message": "Document processed successfully"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
async def query_knowledge_bot(query: str):
    """Query the knowledge bot."""
    try:
        # Generate query embedding
        query_embedding = embedding_generator.generate_single_embedding(query)
        
        # Search for relevant chunks
        relevant_chunks = vector_store.search_vectors(query_embedding)
        
        # Generate response
        response = llm_engine.generate_response(query, relevant_chunks)
        
        return response
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True
    ) 