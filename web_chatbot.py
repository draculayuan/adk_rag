from flask import Flask, render_template, request, jsonify
from google import adk
from typing import List, Dict, Any
from vertexai import agent_engines
import vertexai

app = Flask(__name__)

# Initialize the agent engine
agent_engine = vertexai.agent_engines.get('projects/163097687798/locations/us-central1/reasoningEngines/8537074470983041024')
session = agent_engine.create_session(user_id="test_user")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    query = request.json.get('message', '')
    if not query.strip():
        return jsonify({'response': 'Please enter a message'})
    
    response_text = ""
    for event in agent_engine.stream_query(
        user_id="yuan",
        session_id=session["id"],
        message=query
    ):
        text = event['content']['parts'][0].get('text', None)
        role = event['content']['role']
        if (text is not None) and (role == 'model'):
            response_text += text
    
    return jsonify({'response': response_text})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000) 