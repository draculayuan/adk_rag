from typing import List, Dict, Any
from google.cloud import aiplatform
from vertexai.preview.agents import Agent, AgentConfig
from vertexai.preview.agents.tools import Tool
from ..config import settings

class KnowledgeAgent:
    def __init__(self):
        # Initialize Vertex AI
        aiplatform.init(
            project=settings.GOOGLE_CLOUD_PROJECT,
            location=settings.VERTEX_AI_LOCATION
        )
        
        # Create agent configuration
        self.config = AgentConfig(
            name="cymbal_knowledge_agent",
            description="An agent that helps employees find information about company policies and documents",
            tools=self._create_tools()
        )
        
        # Initialize the agent
        self.agent = Agent(config=self.config)

    def _create_tools(self) -> List[Tool]:
        """Create tools for the agent."""
        return [
            Tool(
                name="search_knowledge_base",
                description="Search through company knowledge base for relevant information",
                function=self._search_knowledge_base
            )
        ]

    def _search_knowledge_base(self, query: str) -> List[Dict[str, Any]]:
        """Search the knowledge base using Vertex AI Vector Search."""
        # This will be handled by Vertex AI Agent Engine's built-in vector search
        pass

    def process_query(self, query: str) -> Dict[str, Any]:
        """Process a user query using the agent."""
        response = self.agent.run(query)
        return {
            "answer": response.text,
            "sources": response.sources
        } 