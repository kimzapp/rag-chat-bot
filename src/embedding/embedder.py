"""
Embedding abstraction layer for the RAG chatbot system.
Provides unified interface for different embedding providers.
"""

from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import List, Dict, Type
import numpy as np
from configs import get_config


@dataclass
class EmbeddingResult:
    text: str
    embedding: np.ndarray


class BaseEmbedder(ABC):
    """Abstract base class for embedders."""
    
    def __init__(self, **kwargs):
        self.config = kwargs
    
    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[EmbeddingResult]:
        """Embed a list of documents."""
        pass
    
    @abstractmethod
    def embed_query(self, text: str) -> EmbeddingResult:
        """Embed a single query."""
        pass
    

class SentenceTransformerEmbedder(BaseEmbedder):
    """Embedder using Sentence Transformers."""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', **kwargs):
        super().__init__(**kwargs)
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: List[str]) -> List[EmbeddingResult]:
        """Embed a list of documents."""
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return [EmbeddingResult(text=text, embedding=emb) for text, emb in zip(texts, embeddings)]

    def embed_query(self, text: str) -> EmbeddingResult:
        """Embed a single query."""
        embedding = self.model.encode(text, convert_to_numpy=True)
        return EmbeddingResult(text=text, embedding=embedding)


class EmbedderFactory:
    """Factory class for creating embedder instances."""
    
    _embedders: Dict[str, Type[BaseEmbedder]] = {
        'sentence-transformer': SentenceTransformerEmbedder,
        # Có thể thêm các embedder khác
        # 'openai': OpenAIEmbedder,
        # 'huggingface': HuggingFaceEmbedder,
    }

    @classmethod
    def create_embedder(cls) -> BaseEmbedder:
        """Create an embedder instance."""
        config = get_config().get_embedding_config()
        provider = config.provider
        model_name = config.model_name
        
        if provider not in cls._embedders:
            raise ValueError(f"Unsupported embedder type: {provider}")

        embedding_config = config.additional_params or {}
        embedding_config['model_name'] = model_name

        embedder_class = cls._embedders[provider]
        return embedder_class(**embedding_config)
