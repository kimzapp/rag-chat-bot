"""
Vector store abstraction layer for the RAG chatbot system.
Provides unified interface for different vector database providers.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Type
from dataclasses import dataclass
import numpy as np

from configs import get_config


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
    
    # @abstractmethod
    # def update_documents(
    #     self,
    #     ids: List[str],
    #     documents: Optional[List[str]] = None,
    #     embeddings: Optional[List[np.ndarray]] = None,
    #     metadatas: Optional[List[Dict[str, Any]]] = None
    # ) -> bool:
    #     """Update existing documents."""
    #     pass
    
    @abstractmethod
    def get_document_count(self) -> int | None:
        """Get total number of documents in the store."""
        pass
    
    @abstractmethod
    def clear_all(self) -> bool:
        """Clear all documents from the store."""
        pass
    
    # @abstractmethod
    # def create_collection(self, collection_name: str) -> bool:
    #     """Create a new collection."""
    #     pass
    
    # @abstractmethod
    # def delete_collection(self, collection_name: str) -> bool:
    #     """Delete a collection."""
    #     pass


class ChromaVectorStore(BaseVectorStore):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _create_chroma_client(self):
        import chromadb
        chromadb_client = chromadb.PersistentClient(**self.config)
        return chromadb_client

    def add_documents(self, documents, embeddings, metadatas, ids):
        try:
            chromadb_client = self._create_chroma_client()
            collection_name = self.config.get('collection_name', 'rag_documents')
            collection = chromadb_client.get_or_create_collection(name=collection_name, embedding_function=None) # No embedding function, we provide embeddings directly
            collection.add(
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )
            return True
        except Exception as e:
            print(f"Error adding documents to ChromaDB: {e}")
            return False

    def search(self, query_embedding, top_k=10, filter_criteria=None):
        try:
            chromadb_client = self._create_chroma_client()
            collection_name = self.config.get('collection_name', 'rag_documents')
            collection = chromadb_client.get_or_create_collection(name=collection_name, embedding_function=None)
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=filter_criteria
            )
            search_results = []
            for doc, meta, score, id_ in zip(results['documents'][0], results['metadatas'][0], results['distances'][0], results['ids'][0]):
                search_results.append(SearchResult(content=doc, metadata=meta, score=score, chunk_id=id_))
            return search_results
        except Exception as e:
            print(f"Error searching in ChromaDB: {e}")
            return []
        
    def delete_documents(self, ids):
        try:
            chromadb_client = self._create_chroma_client()
            collection_name = self.config.get('collection_name', 'rag_documents')
            collection = chromadb_client.get_or_create_collection(name=collection_name, embedding_function=None)
            collection.delete(ids=ids)
            return True
        except Exception as e:
            print(f"Error deleting documents from ChromaDB: {e}")
            return False
        
    def get_document_count(self):
        try:
            chromadb_client = self._create_chroma_client()
            collection_name = self.config.get('collection_name', 'rag_documents')
            collection = chromadb_client.get_or_create_collection(name=collection_name, embedding_function=None)
            return collection.count()
        except Exception as e:
            print(f"Error getting document count from ChromaDB: {e}")
            return None
        
    def clear_all(self):
        try:
            chromadb_client = self._create_chroma_client()
            collection_name = self.config.get('collection_name', 'rag_documents')
            collection = chromadb_client.get_or_create_collection(name=collection_name, embedding_function=None)
            collection.delete()
            return True
        except Exception as e:
            print(f"Error clearing all documents from ChromaDB: {e}")
            return False
        

class VectorStoreFactory:
    """Factory for creating vector store instances."""
    
    _vector_stores: Dict[str, Type[BaseVectorStore]] = {
        'chroma': ChromaVectorStore,
        # 'pinecone': PineconeVectorStore,
        # 'qdrant': QdrantVectorStore,
        # 'weaviate': WeaviateVectorStore,
    }
    
    @classmethod
    def create_vector_store(cls) -> BaseVectorStore:
        """Create vector store based on configuration."""
        config = get_config().get_database_config()  
        provider = config.vector_db_provider
        vector_config = config.vector_db_config.get(provider, {})
        
        if provider not in cls._vector_stores:
            raise ValueError(f"Unsupported vector store provider: {provider}")
        
        vector_store_class = cls._vector_stores[provider]
        return vector_store_class(**vector_config)
    
    # @classmethod
    # def register_vector_store(cls, name: str, vector_store_class: Type[BaseVectorStore]):
    #     """Register a new vector store type."""
    #     cls._vector_stores[name] = vector_store_class


