# agent/tools/__init__.py

"""
The tools package exports all function-tool endpoints for the ADK agent.
"""

from .retrieve import retrieve_documents

__all__ = [
    "retrieve_documents",
]
