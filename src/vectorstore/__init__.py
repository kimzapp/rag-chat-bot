"""
Vector store package for the RAG chatbot system.
Provides abstracted access to different vector database providers.
"""

from .vector_store import (
    SearchResult, BaseVectorStore, ChromaVectorStore, PineconeVectorStore,
    VectorStoreFactory, get_vector_store
)

__all__ = [
    'SearchResult',
    'BaseVectorStore',
    'ChromaVectorStore',
    'PineconeVectorStore',
    'VectorStoreFactory',
    'get_vector_store'
]
