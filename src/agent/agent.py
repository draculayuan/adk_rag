from google.adk.agents import LlmAgent
from vertexai.preview.reasoning_engines import AdkApp

# Import your retrieval tools
from .tools.retrieve import retrieve_documents

# Define the RAG agent using ADK
rag_agent = LlmAgent(
    name="cymbal_knowledge_agent",
    model="gemini-2.0-flash-001",
    description="Company knowledge assistant that can ingest new files on demand and answer based on our internal documents.",
    instruction=(
        "You are a corporate knowledge assistant. "
        "For any question, always call `retrieve_documents(question)` first to fetch relevant context before answering."
    ),
    tools=[retrieve_documents],
)

# Wrap the agent in an AdkApp for deployment
app = AdkApp(agent=rag_agent)

# quick test
for event in app.stream_query(
    user_id="USER_ID",
    message="What is the exchange rate from US dollars to SEK today?",
):
    print(event)