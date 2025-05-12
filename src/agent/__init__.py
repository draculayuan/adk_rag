# agent/__init__.py
"""
Agent package initializer: exposes the AdkApp instance for deployment and the raw rag_agent if needed.
"""
from .agent import app, rag_agent

__all__ = [
    "app",
    "rag_agent",
]