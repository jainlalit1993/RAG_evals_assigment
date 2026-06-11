
"""
LangGraph State Machine
Defines the RAG workflow: Retrieve → Generate
"""
from typing import TypedDict, List
from langchain.schema import Document

class GraphState(TypedDict):
    """
    State schema for the RAG graph
    
    Attributes:
        question: User's input question
        context: Retrieved document chunks
        answer: Generated answer
        sources: Source metadata for citations
    """
    question: str
    context: List[Document]
    answer: str
    sources: List[dict]
