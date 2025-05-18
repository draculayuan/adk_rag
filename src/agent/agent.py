from google.adk.agents import LlmAgent
#from vertexai.preview.reasoning_engines import AdkApp

# Import your retrieval tools
from .tools.retrieve import retrieve_documents

# Define the RAG agent using ADK
rag_agent = LlmAgent(
    name="rag_agent",
    model="gemini-2.0-flash",
    description="An Agent that can provide answer to user questions based on retrieved context",
    instruction=(
        "You are an question answering agent who will answer user's question by retrieving knowledge from the knowledge base by running 'retrieve_documents(query)' function; Please strictly answer user's question based on the knowledge, and remember to cite the source of the knowledge."
    ),
    tools=[retrieve_documents],
)