"""
Vector store abstraction layer for the RAG chatbot system.
Provides unified interface for different vector database providers.
"""

import os
import json
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
import numpy as np

from ..config import get_config


@dataclass
class SearchResult:
    """Represents a search result from vector store."""
    content: str
    metadata: Dict[str, Any]
    score: float
    chunk_id: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'content': self.content,
            'metadata': self.metadata,
            'score': self.score,
            'chunk_id': self.chunk_id
        }


class BaseVectorStore(ABC):
    """Abstract base class for vector stores."""
    
    def __init__(self, **kwargs):
        self.config = kwargs
    
    @abstractmethod
    def add_documents(
        self,
        documents: List[str],
        embeddings: List[np.ndarray],
        metadatas: List[Dict[str, Any]],
        ids: List[str]
    ) -> bool:
        """Add documents with embeddings to the vector store."""
        pass
    
    @abstractmethod
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        filter_criteria: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Search for similar documents."""
        pass
    
    @abstractmethod
    def delete_documents(self, ids: List[str]) -> bool:
        """Delete documents by IDs."""
        pass
    
    @abstractmethod
    def update_documents(
        self,
        ids: List[str],
        documents: Optional[List[str]] = None,
        embeddings: Optional[List[np.ndarray]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> bool:
        """Update existing documents."""
        pass
    
    @abstractmethod
    def get_document_count(self) -> int:
        """Get total number of documents in the store."""
        pass
    
    @abstractmethod
    def clear_all(self) -> bool:
        """Clear all documents from the store."""
        pass
    
    @abstractmethod
    def create_collection(self, collection_name: str) -> bool:
        """Create a new collection."""
        pass
    
    @abstractmethod
    def delete_collection(self, collection_name: str) -> bool:
        """Delete a collection."""
        pass


class VectorStoreFactory:
    """Factory for creating vector store instances."""
    
    def __init__(self):
        self.config = get_config().get_database_config()
    
    def create_vector_store(self) -> BaseVectorStore:
        """Create vector store based on configuration."""
        provider = self.config.vector_db_provider
        vector_config = self.config.vector_db_config.get(provider, {})
        
        if provider == 'chroma':
            # TODO: Implement QdrantVectorStore
            raise Exception("ChromaDb support not yet implemented")
        elif provider == 'pinecone':
            # TODO: Implement QdrantVectorStore
            raise Exception("PipeCone support not yet implemented")
        elif provider == 'qdrant':
            # TODO: Implement QdrantVectorStore
            raise Exception("Qdrant support not yet implemented")
        elif provider == 'weaviate':
            # TODO: Implement WeaviateVectorStore
            raise Exception("Weaviate support not yet implemented")
        else:
            raise Exception(f"Unsupported vector store provider: {provider}")


def get_vector_store() -> BaseVectorStore:
    """Get vector store instance."""
    factory = VectorStoreFactory()
    return factory.create_vector_store()
