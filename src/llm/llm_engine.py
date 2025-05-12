from typing import List, Dict, Any
from google.cloud import aiplatform
from ..config import settings

class LLMEngine:
    def __init__(self):
        self.project = settings.GOOGLE_CLOUD_PROJECT
        self.location = settings.VERTEX_AI_LOCATION
        self.model = settings.LLM_MODEL
        
        # Initialize Vertex AI
        aiplatform.init(
            project=self.project,
            location=self.location
        )
        
        # Initialize the Gemini model
        self.llm = aiplatform.GenerativeModel(self.model)

    def generate_response(self, query: str, context: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate a response using the LLM with retrieved context."""
        # Prepare context
        context_text = self._prepare_context(context)
        
        # Create prompt
        prompt = self._create_prompt(query, context_text)
        
        # Generate response
        response = self.llm.generate_content(prompt)
        
        # Format response with sources
        return {
            "answer": response.text,
            "sources": self._extract_sources(context)
        }

    def _prepare_context(self, context: List[Dict[str, Any]]) -> str:
        """Prepare context for the prompt."""
        context_text = ""
        for item in context:
            context_text += f"Source: {item['metadata']['file_name']}\n"
            context_text += f"Content: {item['text']}\n\n"
        return context_text

    def _create_prompt(self, query: str, context: str) -> str:
        """Create a prompt for the LLM."""
        return f"""You are a helpful AI assistant for the company Cymbal. 
        Use the following context to answer the question. 
        If you cannot find the answer in the context, say so.
        Always cite your sources.

        Context:
        {context}

        Question: {query}

        Answer:"""

    def _extract_sources(self, context: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Extract source information from context."""
        return [
            {
                "file_name": item["metadata"]["file_name"],
                "chunk_index": item["metadata"]["chunk_index"]
            }
            for item in context
        ] 