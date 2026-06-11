
"""
RAG Agent Implementation
Handles retrieval and answer generation
"""
import os
import yaml
from typing import List
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
from langgraph.graph import StateGraph, END
from app.graph import GraphState
from colorama import Fore, Style

# Load environment
load_dotenv()

class RAGAgent:
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the RAG agent"""
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.config['embeddings']['model_name']
        )
        
        # Load vectorstore
        self.vectorstore = Chroma(
            collection_name=self.config['vectordb']['collection_name'],
            embedding_function=self.embeddings,
            persist_directory=self.config['vectordb']['persist_directory']
        )
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model=self.config['llm']['model'],
            temperature=self.config['llm']['temperature']
        )
        
        # Define prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful healthcare insurance policy assistant.

Answer the user's question using ONLY the information provided in the context below.

RULES:
1. If the answer is in the context, provide a clear and concise response.
2. If the answer is NOT in the context, respond with: "I don't have information about that in the provided documents."
3. Do NOT make up information or use external knowledge.
4. Cite the source document when answering (e.g., "According to [source]...")
5. Be specific and helpful.

Context:
{context}
"""),
            ("human", "{question}")
        ])
    
    def retrieve_node(self, state: GraphState) -> GraphState:
        """
        Retrieve relevant documents from vector database
        """
        question = state["question"]
        
        # Perform similarity search
        docs = self.vectorstore.similarity_search(
            question,
            k=self.config['retrieval']['top_k']
        )
        
        # Extract source information
        sources = []
        for doc in docs:
            sources.append({
                "source": doc.metadata.get("source", "Unknown"),
                "page": doc.metadata.get("page", "N/A")
            })
        
        # Update state
        state["context"] = docs
        state["sources"] = sources
        
        return state
    
    def generate_node(self, state: GraphState) -> GraphState:
        """
        Generate answer using retrieved context
        """
        question = state["question"]
        context = state["context"]
        
        # Check if we have context
        if not context:
            state["answer"] = "I don't have any relevant information in the database to answer this question."
            return state
        
        # Format context
        context_text = "\n\n".join([
            f"[Source: {doc.metadata.get('source', 'Unknown')}, Page: {doc.metadata.get('page', 'N/A')}]\n{doc.page_content}"
            for doc in context
        ])
        
        # Generate answer
        chain = self.prompt | self.llm
        response = chain.invoke({
            "context": context_text,
            "question": question
        })
        
        state["answer"] = response.content
        
        return state
    
    def build_graph(self):
        """
        Build the LangGraph workflow
        
        Graph structure:
        START → retrieve → generate → END
        """
        # Create graph
        workflow = StateGraph(GraphState)
        
        # Add nodes
        workflow.add_node("retrieve", self.retrieve_node)
        workflow.add_node("generate", self.generate_node)
        
        # Add edges
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", END)
        
        # Compile
        graph = workflow.compile()
        
        return graph
    
    def query(self, question: str) -> dict:
        """
        Query the RAG system
        
        Args:
            question: User's question
            
        Returns:
            dict with answer and sources
        """
        # Build graph
        graph = self.build_graph()
        
        # Run graph
        result = graph.invoke({
            "question": question,
            "context": [],
            "answer": "",
            "sources": []
        })
        
        return {
            "question": question,
            "answer": result["answer"],
            "sources": result["sources"]
        }
