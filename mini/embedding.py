"""
Embedding module for generating embeddings using OpenAI-compatible APIs.
Supports OpenAI, Azure OpenAI, and any other OpenAI-compatible endpoints.
"""

import os
from typing import List, Optional, Union

from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class EmbeddingConfig(BaseModel):
    """Configuration for the embedding model."""
    
    api_key: str = Field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY", ""),
        description="API key for the embedding service"
    )
    base_url: Optional[str] = Field(
        default_factory=lambda: os.getenv("OPENAI_BASE_URL"),
        description="Base URL for OpenAI-compatible API (e.g., 'https://api.openai.com/v1')"
    )
    model: str = Field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        description="Name of the embedding model to use"
    )
    dimensions: Optional[int] = Field(
        default=None,
        description="Number of dimensions for the embedding (if supported by the model)"
    )
    timeout: float = Field(
        default=60.0,
        description="Timeout for API requests in seconds"
    )
    max_retries: int = Field(
        default=3,
        description="Maximum number of retries for failed requests"
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)


class EmbeddingModel:
    """
    A flexible embedding class that works with any OpenAI-compatible API.
    
    Examples:
        # Using OpenAI
        model = EmbeddingModel(
            api_key="sk-...",
            model="text-embedding-3-small"
        )
        
        # Using Azure OpenAI
        model = EmbeddingModel(
            api_key="...",
            base_url="https://your-resource.openai.azure.com/openai/deployments/your-deployment",
            model="text-embedding-ada-002"
        )
        
        # Using local model (e.g., llama.cpp server)
        model = EmbeddingModel(
            api_key="not-needed",
            base_url="http://localhost:8080/v1",
            model="text-embedding"
        )
        
        # Generate embeddings
        embedding = model.embed("Hello, world!")
        embeddings = model.embed_batch(["Text 1", "Text 2", "Text 3"])
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        dimensions: Optional[int] = None,
        timeout: float = 60.0,
        max_retries: int = 3,
        config: Optional[EmbeddingConfig] = None,
    ):
        """
        Initialize the embedding model.
        
        Args:
            api_key: API key for the service (falls back to OPENAI_API_KEY env var)
            base_url: Base URL for the API (falls back to OPENAI_BASE_URL env var)
            model: Model name (falls back to EMBEDDING_MODEL env var or default)
            dimensions: Number of dimensions for embeddings (if supported)
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries
            config: Pre-configured EmbeddingConfig object (overrides other params)
        """
        if config:
            self.config = config
        else:
            self.config = EmbeddingConfig(
                api_key=api_key or os.getenv("OPENAI_API_KEY", ""),
                base_url=base_url or os.getenv("OPENAI_BASE_URL"),
                model=model or os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
                dimensions=dimensions,
                timeout=timeout,
                max_retries=max_retries
            )
        
        # Validate API key
        if not self.config.api_key:
            raise ValueError(
                "API key is required. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        # Initialize OpenAI client
        client_kwargs = {
            "api_key": self.config.api_key,
            "timeout": self.config.timeout,
            "max_retries": self.config.max_retries,
        }
        
        if self.config.base_url:
            client_kwargs["base_url"] = self.config.base_url
        
        self.client = OpenAI(**client_kwargs)
    
    def embed_chunks(self, chunks: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of chunks.
        
        Args:
            chunks: List of input chunks to embed
            batch_size: Number of chunks to process in each API call
            **kwargs: Additional parameters to pass to the API
        """
        if not chunks:
            raise ValueError("Input texts list cannot be empty")

        chunks = [chunk.text for chunk in chunks]
        try:
            response = self.client.embeddings.create(
                input=chunks,
                model=self.config.model
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            raise ValueError(f"Error generating embeddings: {e}")


    def embed_query(self, query: str) -> List[float]:
        """
        Generate an embedding for a query.
        
        Args:
            query: Query string to embed
        """
        try:
            response = self.client.embeddings.create(
                input=query,
                model=self.config.model
            )
            return response.data[0].embedding
        except Exception as e:
            raise ValueError(f"Error generating embedding: {e}")

    def __repr__(self) -> str:
        """String representation of the embedding model."""
        base_url = self.config.base_url or "default"
        return (
            f"EmbeddingModel(model='{self.config.model}', "
            f"base_url='{base_url}', "
            f"dimensions={self.config.dimensions})"
        )


if __name__ == "__main__":
    from loader import DocumentLoader
    from chunker import Chunker
    loader = DocumentLoader()
    document = loader.load("./mini/documents/budget_speech.pdf")
    chunker = Chunker()
    chunks = chunker.chunk(document)
    embedding_model = EmbeddingModel()
    embeddings = embedding_model.embed_chunks(chunks)
    print(len(embeddings))
