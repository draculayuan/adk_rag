from google import adk
#from vertexai.preview.reasoning_engines import AdkApp
from typing import List, Dict, Any
from vertexai import agent_engines
import vertexai


agent_engine = vertexai.agent_engines.get('projects/163097687798/locations/us-central1/reasoningEngines/2227531393036976128')

#retrieve session
session = agent_engine.get_session(user_id="yuan", session_id = '5569172079976120320')

#query
query = "how did you managed to keep track of what we discussed?"
for event in agent_engine.stream_query(
    user_id="yuan",
    session_id=session["id"],  # Optional
    message=query
):
    
    text = event['content']['parts'][0].get('text', None)
    role = event['content']['role']
    if (text is not None) and (role == 'model'):
        print(text)