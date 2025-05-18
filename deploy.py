from vertexai import agent_engines
import vertexai

from src.agent.agent import rag_agent

vertexai.init(staging_bucket="gs://yuan_evernote_rag_cs")

remote_app = agent_engines.create(
    agent_engine=rag_agent,
    requirements=[
        "google-cloud-aiplatform[agent_engines,adk,langchain,ag2,llama_index]==1.90.0",
        "google-cloud-firestore==2.20.2"
    ],
    extra_packages=["agent"]
)

