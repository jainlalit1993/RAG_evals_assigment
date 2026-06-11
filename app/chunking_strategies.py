"""
Advanced Chunking Strategies for RAG System
Includes: Token-based, Semantic, Agentic, and Recursive chunking
"""
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    TokenTextSplitter
)
from langchain_openai import ChatOpenAI
from typing import List, Optional
from langchain.schema import Document
import os
from dotenv import load_dotenv

load_dotenv()


class AdvancedChunkingStrategies:
    """
    Collection of advanced chunking strategies
    """
    
    @staticmethod
    def token_based_with_overlap(
        chunk_size: int = 512, 
        chunk_overlap: int = 128,
        encoding_name: str = "cl100k_base"
    ):
        """
        Token-based chunking with overlap
        
        Uses tiktoken encoding to count tokens (same as GPT models)
        
        Args:
            chunk_size: Number of tokens per chunk (default: 512)
            chunk_overlap: Number of overlapping tokens (default: 128)
            encoding_name: Tiktoken encoding ("cl100k_base" for GPT-3.5/4)
        
        Best for:
            - Precise token control for LLM context windows
            - Aligning chunks with model token limits
            - Preventing token overflow errors
        
        Example:
            - GPT-3.5-turbo context: 4096 tokens
            - Chunk size: 512 tokens = ~8 chunks per context
            - Overlap ensures context continuity
        """
        return TokenTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            encoding_name=encoding_name
        )
    
    @staticmethod
    def semantic_chunking(
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: Optional[List[str]] = None
    ):
        """
        Semantic chunking - respects natural language boundaries
        
        Splits text by semantic units (paragraphs → sentences → words)
        
        Args:
            chunk_size: Target chunk size in characters
            chunk_overlap: Character overlap between chunks
            separators: Custom separators (defaults to semantic boundaries)
        
        Best for:
            - Natural language documents
            - Preserving meaning and context
            - General-purpose RAG
        
        How it works:
            1. Try splitting by paragraphs (\n\n)
            2. If chunk too large, split by sentences (. )
            3. If still too large, split by words ( )
            4. Last resort: split by characters
        """
        if separators is None:
            separators = [
                "\n\n\n",    # Section breaks
                "\n\n",      # Paragraphs
                "\n",        # Lines
                ". ",        # Sentences
                "! ",        # Exclamations
                "? ",        # Questions
                "; ",        # Semicolons
                ", ",        # Commas
                " ",         # Words
                ""           # Characters
            ]
        
        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
            length_function=len,
            is_separator_regex=False,
            keep_separator=True
        )
    
    @staticmethod
    def agentic_chunking(
        llm: Optional[ChatOpenAI] = None,
        max_chunk_size: int = 1500,
        overlap: int = 200
    ):
        """
        Agentic chunking - uses LLM to intelligently split content
        
        LLM analyzes content and creates semantically coherent chunks
        
        Args:
            llm: ChatOpenAI instance (uses GPT-3.5-turbo if None)
            max_chunk_size: Maximum chunk size in characters
            overlap: Character overlap between chunks
        
        Best for:
            - Complex documents with varied structure
            - Documents requiring context-aware splitting
            - High-quality retrieval (worth the extra LLM calls)
        
        How it works:
            1. LLM reads the text
            2. Identifies logical breaking points
            3. Creates chunks that preserve semantic meaning
            4. Adds overlap for context continuity
        
        Note: This is slower and costs API tokens, but produces
              the highest quality chunks for complex content.
        """
        if llm is None:
            llm = ChatOpenAI(
                model="gpt-4.1-mini",
                temperature=0
            )
        
        return AgenticChunker(
            llm=llm,
            max_chunk_size=max_chunk_size,
            overlap=overlap
        )
    
    @staticmethod
    def recursive_chunking(
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: Optional[List[str]] = None
    ):
        """
        Recursive chunking - hierarchical text splitting
        
        Recursively splits text trying each separator in order
        
        Args:
            chunk_size: Target chunk size in characters
            chunk_overlap: Character overlap between chunks
            separators: List of separators to try (in order)
        
        Best for:
            - Hierarchical documents (markdown, code, structured text)
            - Documents with nested structure
            - Preserving document hierarchy
        
        How it works:
            1. Try first separator (e.g., "\n\n\n" for sections)
            2. If chunks still too large, try next separator
            3. Recurse until chunks are appropriate size
            4. Maintains document structure
        
        Example separators for code:
            ["\n\nclass ", "\n\ndef ", "\n\n", "\n", " "]
        
        Example separators for markdown:
            ["# ", "## ", "### ", "\n\n", "\n", ". "]
        """
        if separators is None:
            # Default: general text
            separators = ["\n\n", "\n", ". ", "! ", "? ", " ", ""]
        
        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
            length_function=len,
            is_separator_regex=False
        )


class AgenticChunker:
    """
    LLM-powered intelligent chunking
    Uses GPT to identify optimal chunk boundaries
    """
    
    def __init__(self, llm: ChatOpenAI, max_chunk_size: int = 1500, overlap: int = 200):
        self.llm = llm
        self.max_chunk_size = max_chunk_size
        self.overlap = overlap
    
    def split_text(self, text: str) -> List[str]:
        """Split text using LLM intelligence"""
        
        # For very short text, return as-is
        if len(text) <= self.max_chunk_size:
            return [text]
        
        # Use LLM to identify chunk boundaries
        prompt = f"""You are an expert at splitting text into semantically coherent chunks.

Given the following text, identify the best places to split it into chunks of approximately {self.max_chunk_size} characters.

RULES:
1. Each chunk should be a complete, coherent unit of meaning
2. Split at natural boundaries (paragraphs, sections, topics)
3. Aim for chunks around {self.max_chunk_size} characters (can be ±20%)
4. Preserve context - don't split in the middle of important information

Return ONLY the split points as character indices (comma-separated numbers).
Example: 450,920,1350

Text to split:
{text[:5000]}

Split points (character indices):"""

        try:
            response = self.llm.invoke(prompt)
            split_indices = [int(x.strip()) for x in response.content.strip().split(",")]
            
            # Create chunks based on LLM's suggested splits
            chunks = []
            start = 0
            
            for idx in split_indices:
                if start < len(text):
                    # Add overlap from previous chunk
                    chunk_start = max(0, start - self.overlap)
                    chunk = text[chunk_start:idx].strip()
                    if chunk:
                        chunks.append(chunk)
                    start = idx
            
            # Add final chunk
            if start < len(text):
                chunk_start = max(0, start - self.overlap)
                final_chunk = text[chunk_start:].strip()
                if final_chunk:
                    chunks.append(final_chunk)
            
            return chunks if chunks else [text]
        
        except Exception as e:
            # Fallback to semantic chunking if LLM fails
            print(f"Agentic chunking failed, falling back to semantic: {e}")
            fallback = AdvancedChunkingStrategies.semantic_chunking(
                chunk_size=self.max_chunk_size,
                chunk_overlap=self.overlap
            )
            return fallback.split_text(text)
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents using LLM intelligence"""
        chunked_docs = []
        
        for doc in documents:
            chunks = self.split_text(doc.page_content)
            
            for i, chunk in enumerate(chunks):
                chunked_doc = Document(
                    page_content=chunk,
                    metadata={
                        **doc.metadata,
                        "chunk_index": i,
                        "total_chunks": len(chunks)
                    }
                )
                chunked_docs.append(chunked_doc)
        
        return chunked_docs


def compare_all_strategies(text: str) -> dict:
    """
    Compare all chunking strategies on sample text
    
    Args:
        text: Sample text to analyze
        
    Returns:
        Dictionary with comparison results
    """
    results = {}
    
    # 1. Token-based with overlap
    token_chunker = AdvancedChunkingStrategies.token_based_with_overlap(
        chunk_size=512,
        chunk_overlap=128
    )
    token_chunks = token_chunker.split_text(text)
    results["Token-based (512 tokens, 128 overlap)"] = {
        "chunks": len(token_chunks),
        "avg_size": sum(len(c) for c in token_chunks) // len(token_chunks) if token_chunks else 0,
        "first_chunk": token_chunks[0][:150] if token_chunks else ""
    }
    
    # 2. Semantic chunking
    semantic_chunker = AdvancedChunkingStrategies.semantic_chunking(
        chunk_size=1000,
        chunk_overlap=200
    )
    semantic_chunks = semantic_chunker.split_text(text)
    results["Semantic (1000 chars, 200 overlap)"] = {
        "chunks": len(semantic_chunks),
        "avg_size": sum(len(c) for c in semantic_chunks) // len(semantic_chunks) if semantic_chunks else 0,
        "first_chunk": semantic_chunks[0][:150] if semantic_chunks else ""
    }
    
    # 3. Recursive chunking
    recursive_chunker = AdvancedChunkingStrategies.recursive_chunking(
        chunk_size=1000,
        chunk_overlap=200
    )
    recursive_chunks = recursive_chunker.split_text(text)
    results["Recursive (1000 chars, 200 overlap)"] = {
        "chunks": len(recursive_chunks),
        "avg_size": sum(len(c) for c in recursive_chunks) // len(recursive_chunks) if recursive_chunks else 0,
        "first_chunk": recursive_chunks[0][:150] if recursive_chunks else ""
    }
    
    # 4. Agentic chunking (skip if text too short or no API key)
    if len(text) > 500 and os.getenv("OPENAI_API_KEY"):
        try:
            agentic_chunker = AdvancedChunkingStrategies.agentic_chunking(
                max_chunk_size=1500,
                overlap=200
            )
            agentic_chunks = agentic_chunker.split_text(text)
            results["Agentic (LLM-powered, 1500 chars)"] = {
                "chunks": len(agentic_chunks),
                "avg_size": sum(len(c) for c in agentic_chunks) // len(agentic_chunks) if agentic_chunks else 0,
                "first_chunk": agentic_chunks[0][:150] if agentic_chunks else ""
            }
        except Exception as e:
            results["Agentic (LLM-powered, 1500 chars)"] = {
                "chunks": 0,
                "avg_size": 0,
                "first_chunk": f"Error: {str(e)}"
            }
    
    return results