#from google import adk
#from google.adk.agents import LlmAgent
#from vertexai.preview.reasoning_engines import AdkApp
#from typing import List, Dict, Any
#from google.adk.tools import ToolContext
#from google.adk.sessions import InMemorySessionService
from vertexai import agent_engines
#from google.adk.tools import ToolContext
import vertexai

from agent.agent import rag_agent

#app_agent = AdkApp(agent=rag_agent)

vertexai.init(staging_bucket="gs://yuan_evernote_rag_cs")

remote_app = agent_engines.create(
    agent_engine=rag_agent,
    #requirements="requirements_agent.txt",
    requirements=[
        "google-cloud-aiplatform[agent_engines,adk,langchain,ag2,llama_index]==1.90.0",
        "google-cloud-firestore==2.20.2"
    ],
    extra_packages=["agent"]
)

