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
        self.logger = get_logger(self.__class__.__name__)
    
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


class ChromaVectorStore(BaseVectorStore):
    """ChromaDB vector store implementation."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.persist_directory = kwargs.get('persist_directory', 'data/chroma_db')
        self.collection_name = kwargs.get('collection_name', 'rag_documents')
        self.distance_metric = kwargs.get('distance_metric', 'cosine')
        self.client = None
        self.collection = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize ChromaDB client."""
        try:
            import chromadb
            from chromadb.config import Settings
            
            # Create client with persistence
            self.client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(anonymized_telemetry=False)
            )
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": self.distance_metric}
            )
            
            self.logger.info(f"ChromaDB initialized with collection: {self.collection_name}")
            
        except ImportError:
            raise VectorStoreError(
                "ChromaDB not installed. Install with: pip install chromadb",
                error_code=ErrorCode.VECTOR_STORE_CONNECTION_ERROR
            )
        except Exception as e:
            raise VectorStoreError(
                f"Failed to initialize ChromaDB: {e}",
                error_code=ErrorCode.VECTOR_STORE_CONNECTION_ERROR
            )
    
    @retry_on_error(max_retries=3, delay=1.0)
    @handle_exceptions(raise_on_error=True)
    def add_documents(
        self,
        documents: List[str],
        embeddings: List[np.ndarray],
        metadatas: List[Dict[str, Any]],
        ids: List[str]
    ) -> bool:
        """Add documents to ChromaDB."""
        try:
            # Convert numpy arrays to lists
            embeddings_list = [emb.tolist() for emb in embeddings]
            
            # Convert metadata values to strings for ChromaDB compatibility
            processed_metadatas = []
            for metadata in metadatas:
                processed_metadata = {}
                for key, value in metadata.items():
                    if isinstance(value, (dict, list)):
                        processed_metadata[key] = json.dumps(value)
                    else:
                        processed_metadata[key] = str(value)
                processed_metadatas.append(processed_metadata)
            
            self.collection.add(
                documents=documents,
                embeddings=embeddings_list,
                metadatas=processed_metadatas,
                ids=ids
            )
            
            self.logger.info(f"Added {len(documents)} documents to ChromaDB")
            return True
            
        except Exception as e:
            raise VectorStoreError(
                f"Failed to add documents to ChromaDB: {e}",
                error_code=ErrorCode.VECTOR_STORE_INDEX_ERROR
            )
    
    @retry_on_error(max_retries=3, delay=1.0)
    @handle_exceptions(raise_on_error=True)
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        filter_criteria: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Search ChromaDB for similar documents."""
        try:
            # Convert numpy array to list
            query_embedding_list = query_embedding.tolist()
            
            # Build where clause for filtering
            where_clause = None
            if filter_criteria:
                where_clause = {}
                for key, value in filter_criteria.items():
                    if isinstance(value, (dict, list)):
                        where_clause[key] = json.dumps(value)
                    else:
                        where_clause[key] = str(value)
            
            # Perform search
            results = self.collection.query(
                query_embeddings=[query_embedding_list],
                n_results=top_k,
                where=where_clause
            )
            
            # Convert results to SearchResult objects
            search_results = []
            
            if results['documents'] and results['documents'][0]:
                documents = results['documents'][0]
                metadatas = results['metadatas'][0]
                distances = results['distances'][0] if results['distances'] else [0.0] * len(documents)
                ids = results['ids'][0]
                
                for doc, metadata, distance, doc_id in zip(documents, metadatas, distances, ids):
                    # Convert distance to similarity score (1 - distance for cosine)
                    score = 1.0 - distance if distance is not None else 0.0
                    
                    # Parse JSON metadata back
                    parsed_metadata = {}
                    for key, value in metadata.items():
                        try:
                            parsed_metadata[key] = json.loads(value)
                        except (json.JSONDecodeError, TypeError):
                            parsed_metadata[key] = value
                    
                    search_results.append(SearchResult(
                        content=doc,
                        metadata=parsed_metadata,
                        score=score,
                        chunk_id=doc_id
                    ))
            
            return search_results
            
        except Exception as e:
            raise VectorStoreError(
                f"Failed to search ChromaDB: {e}",
                error_code=ErrorCode.VECTOR_STORE_SEARCH_ERROR
            )
    
    @handle_exceptions(raise_on_error=True)
    def delete_documents(self, ids: List[str]) -> bool:
        """Delete documents from ChromaDB."""
        try:
            self.collection.delete(ids=ids)
            self.logger.info(f"Deleted {len(ids)} documents from ChromaDB")
            return True
        except Exception as e:
            raise VectorStoreError(
                f"Failed to delete documents from ChromaDB: {e}",
                error_code=ErrorCode.VECTOR_STORE_INDEX_ERROR
            )
    
    @handle_exceptions(raise_on_error=True)
    def update_documents(
        self,
        ids: List[str],
        documents: Optional[List[str]] = None,
        embeddings: Optional[List[np.ndarray]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> bool:
        """Update documents in ChromaDB."""
        try:
            update_data = {'ids': ids}
            
            if documents:
                update_data['documents'] = documents
            
            if embeddings:
                update_data['embeddings'] = [emb.tolist() for emb in embeddings]
            
            if metadatas:
                processed_metadatas = []
                for metadata in metadatas:
                    processed_metadata = {}
                    for key, value in metadata.items():
                        if isinstance(value, (dict, list)):
                            processed_metadata[key] = json.dumps(value)
                        else:
                            processed_metadata[key] = str(value)
                    processed_metadatas.append(processed_metadata)
                update_data['metadatas'] = processed_metadatas
            
            self.collection.update(**update_data)
            self.logger.info(f"Updated {len(ids)} documents in ChromaDB")
            return True
            
        except Exception as e:
            raise VectorStoreError(
                f"Failed to update documents in ChromaDB: {e}",
                error_code=ErrorCode.VECTOR_STORE_INDEX_ERROR
            )
    
    def get_document_count(self) -> int:
        """Get document count from ChromaDB."""
        try:
            return self.collection.count()
        except Exception as e:
            self.logger.error(f"Failed to get document count: {e}")
            return 0
    
    @handle_exceptions(raise_on_error=True)
    def clear_all(self) -> bool:
        """Clear all documents from ChromaDB."""
        try:
            # Delete the collection and recreate it
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": self.distance_metric}
            )
            self.logger.info("Cleared all documents from ChromaDB")
            return True
        except Exception as e:
            raise VectorStoreError(
                f"Failed to clear ChromaDB: {e}",
                error_code=ErrorCode.VECTOR_STORE_INDEX_ERROR
            )
    
    def create_collection(self, collection_name: str) -> bool:
        """Create a new collection in ChromaDB."""
        try:
            self.client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": self.distance_metric}
            )
            return True
        except Exception as e:
            self.logger.error(f"Failed to create collection {collection_name}: {e}")
            return False
    
    def delete_collection(self, collection_name: str) -> bool:
        """Delete a collection from ChromaDB."""
        try:
            self.client.delete_collection(collection_name)
            return True
        except Exception as e:
            self.logger.error(f"Failed to delete collection {collection_name}: {e}")
            return False


class VectorStoreFactory:
    """Factory for creating vector store instances."""
    
    def __init__(self):
        self.config = get_config().get_database_config()
    
    def create_vector_store(self) -> BaseVectorStore:
        """Create vector store based on configuration."""
        provider = self.config.vector_db_provider
        vector_config = self.config.vector_db_config.get(provider, {})
        
        if provider == 'chroma':
            return ChromaVectorStore(**vector_config)
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
