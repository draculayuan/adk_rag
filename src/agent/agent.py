from google.adk.agents import LlmAgent

# from vertexai.preview.reasoning_engines import AdkApp

# Import your retrieval tools
from .tools.retrieve import retrieve_documents

# Define the RAG agent using ADK
rag_agent = LlmAgent(
    name="rag_agent",
    model="gemini-2.5-pro-preview-05-06",
    description="An Agent that can provide answer to user questions based on retrieved context",
    instruction=(
        """
        You are an question answering agent who will answer user's question by retrieving knowledge from the knowledge base by running 'retrieve_documents(query)' function; Please strictly answer user's question based on the knowledge, and remember to cite the source of the knowledge.\n
        
        When answering, you MUST cite your sources using citation numbers in square brackets [1], [2], etc. Each citation number corresponds to the relevant context item in the order it was provided. \n
                For example, if you're using information from the first context item, cite it as [1]. If you're using information from multiple context items, cite them as [1][2]. At the end of your answer, you must also add a 'Reference' section to provide the 'file_path' of the context you have cited.  \n
                E.g., vPost provides you with personalized delivery addresses in 8 countries [2]. References: [2] data/vpost_faqs.csv
        """
    ),
    tools=[retrieve_documents],
)
