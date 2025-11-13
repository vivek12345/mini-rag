"""
Mini RAG - A lightweight RAG (Retrieval-Augmented Generation) library.

Provides document loading, chunking, embedding, vector storage, retrieval, and re-ranking
with built-in observability support.
"""

from mini.loader import DocumentLoader
from mini.chunker import Chunker
from mini.embedding import EmbeddingModel, EmbeddingConfig
from mini.store import VectorStore, MilvusConfig
from mini.reranker import (
    BaseReranker,
    RerankResult,
    CohereReranker,
    SentenceTransformerReranker,
    LLMReranker,
    create_reranker,
)
from mini.observability import LangfuseConfig
from mini.rag import (
    AgenticRAG,
    LLMConfig,
    RetrievalConfig,
    RerankerConfig,
)

__version__ = "0.1.0"

__all__ = [
    # Core classes
    "DocumentLoader",
    "Chunker",
    "EmbeddingModel",
    "VectorStore",
    "AgenticRAG",
    
    # Configurations
    "EmbeddingConfig",
    "MilvusConfig",
    "LLMConfig",
    "RetrievalConfig",
    "RerankerConfig",
    "LangfuseConfig",
    
    # Rerankers
    "BaseReranker",
    "RerankResult",
    "CohereReranker",
    "SentenceTransformerReranker",
    "LLMReranker",
    "create_reranker",
    
    # Version
    "__version__",
]

