"""
Vector store module for managing Milvus vector database operations.
"""

import os
from typing import List, Optional, Dict, Any
from pymilvus import (
    MilvusClient,
    AsyncMilvusClient,
    connections,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType,
    utility,
    AnnSearchRequest,
    RRFRanker,
    Function,
    FunctionType
)
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from mini.embedding import EmbeddingConfig
from mini.logger import logger

# Load environment variables
load_dotenv()


class MilvusConfig(BaseModel):
    """Configuration for Milvus vector store."""
    
    uri: str = Field(
        default_factory=lambda: os.getenv("MILVUS_URI"),
        description="Milvus server URI"
    )
    token: str = Field(
        default_factory=lambda: os.getenv("MILVUS_TOKEN"),
        description="Milvus server token"
    )
    collection_name: str = Field(
        default="documents",
        description="Name of the collection to use"
    )
    dimension: int = Field(
        default=1536,
        description="Dimension of the embeddings (e.g., 1536 for text-embedding-3-small)"
    )
    metric_type: str = Field(
        default="IP",
        description="Metric type for similarity search (IP, L2, COSINE)"
    )
    index_type: str = Field(
        default="IVF_FLAT",
        description="Index type for vector search"
    )
    nlist: int = Field(
        default=128,
        description="Number of cluster units for IVF index"
    )


class VectorStore:
    """
    A vector store class for managing embeddings in Milvus.
    
    Examples:
        # Initialize and connect
        store = VectorStore(
            host="localhost",
            port="19530",
            collection_name="my_documents",
            dimension=1536
        )
        
        # Insert embeddings with metadata
        store.insert(
            embeddings=[[0.1, 0.2, ...], [0.3, 0.4, ...]],
            texts=["Text 1", "Text 2"],
            metadata=[{"source": "doc1.pdf"}, {"source": "doc2.pdf"}]
        )
        
        # Search for similar vectors
        results = store.search(
            query_embedding=[0.1, 0.2, ...],
            top_k=5
        )
    """
    
    def __init__(
        self,
        uri: str,
        token: str,
        collection_name: Optional[str] = None,
        dimension: Optional[int] = None,
        metric_type: str = "IP",
        index_type: str = "IVF_FLAT",
        nlist: int = 128,
        config: Optional[MilvusConfig] = None,
    ):
        """
        Initialize the Milvus vector store.
        
        Args:
            host: Milvus server host (defaults to MILVUS_HOST env var or 'localhost')
            port: Milvus server port (defaults to MILVUS_PORT env var or '19530')
            collection_name: Name of the collection (defaults to 'documents')
            dimension: Embedding dimension (defaults to 1536)
            metric_type: Similarity metric (IP, L2, COSINE)
            index_type: Index type for search (IVF_FLAT, IVF_SQ8, HNSW, etc.)
            nlist: Number of cluster units for IVF index
            config: Pre-configured MilvusConfig object
        """
        if config:
            self.config = config
        else:   
            self.config = MilvusConfig(
                uri=uri,
                token=token,
                collection_name=collection_name,
                dimension=dimension,
                metric_type=metric_type,
                index_type=index_type,
                nlist=nlist,
            )
        
        self.__connect()
        self.collection: Optional[Collection] = None
        self._create_or_load_collection()

    def __connect(self):
        """Connect to Milvus."""
        connections.connect(uri=self.config.uri, token=self.config.token)
        logger.info(f"✅ Connected to Milvus at {self.config.uri}")

    def _create_or_load_collection(self):
        """Create a new collection or load an existing one."""
        if utility.has_collection(self.config.collection_name):
            # Load existing collection
            self.collection = Collection(self.config.collection_name)
            self.collection.load()
            logger.info(f"✅ Loaded existing collection: {self.config.collection_name}")
        else:
            # Create new collection
            self._create_collection()
            logger.info(f"✅ Created new collection: {self.config.collection_name}")
    
    def _create_collection(self):
        """Create a new collection with the specified schema."""
        # Define schema
        analyzer_params = {"tokenizer": "standard", "filter": ["lowercase"]}
        fields = [
            FieldSchema(
                name="id",
                dtype=DataType.INT64,
                is_primary=True,
                auto_id=True,
                description="Primary ID"
            ),
            FieldSchema(
                name="embedding",
                dtype=DataType.FLOAT_VECTOR,
                dim=self.config.dimension,
                description="Embedding vector"
            ),
            FieldSchema(
                name="sparse_vector",
                dtype=DataType.SPARSE_FLOAT_VECTOR,
                description="Sparse embedding vector"
            ),
            FieldSchema(
                name="text",
                dtype=DataType.VARCHAR,
                max_length=65535,
                description="Original text",
                analyzer_params=analyzer_params,
                enable_match=True,  # Enable text matching
                enable_analyzer=True,  # Enable text analysis
            ),
            FieldSchema(
                name="metadata",
                dtype=DataType.JSON,
                description="Metadata as JSON"
            ),
        ]
        
        schema = CollectionSchema(
            fields=fields,
            description=f"Collection for {self.config.collection_name}"
        )

        # Define BM25 function to generate sparse vectors from text
        bm25_function = Function(
            name="bm25",
            function_type=FunctionType.BM25,
            input_field_names=["text"],
            output_field_names="sparse_vector",
        )

        # Add the function to schema
        schema.add_function(bm25_function)
        
        # Create collection
        self.collection = Collection(
            name=self.config.collection_name,
            schema=schema
        )
        
        # Create index
        index_params = {
            "metric_type": self.config.metric_type,
            "index_type": self.config.index_type,
            "params": {"nlist": self.config.nlist}
        }
        
        self.collection.create_index(
            field_name="embedding",
            index_params=index_params
        )

        self.collection.create_index(
            field_name="sparse_vector",
            index_params={
                "index_type": "SPARSE_INVERTED_INDEX",
                "metric_type": "BM25"
            }
        )
        
        # Load collection to memory
        self.collection.load()
    
    def insert(
        self,
        embeddings: List[List[float]],
        texts: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> List[int]:
        """
        Insert embeddings with associated texts and metadata.
        
        Args:
            embeddings: List of embedding vectors
            texts: List of text strings corresponding to embeddings
            metadata: Optional list of metadata dictionaries
        
        Returns:
            List of inserted IDs
        """
        if not embeddings or not texts:
            raise ValueError("Embeddings and texts cannot be empty")
        
        if len(embeddings) != len(texts):
            raise ValueError("Embeddings and texts must have the same length")
        
        # Prepare metadata
        if metadata is None:
            metadata = [{}] * len(embeddings)
        elif len(metadata) != len(embeddings):
            raise ValueError("Metadata must have the same length as embeddings")
        
        # Insert data
        entities = [
            embeddings,
            texts,
            metadata
        ]
        
        result = self.collection.insert(entities)
        self.collection.flush()
        
        logger.info(f"✅ Inserted {len(embeddings)} vectors into {self.config.collection_name}")
        return result.primary_keys
    
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filter_expr: Optional[str] = None,
        output_fields: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors.
        
        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            filter_expr: Optional filter expression (e.g., 'metadata["source"] == "doc1.pdf"')
            output_fields: List of fields to return in results
        
        Returns:
            List of search results with scores and metadata
        """
        if output_fields is None:
            output_fields = ["text", "metadata"]
        
        search_params = {
            "metric_type": self.config.metric_type,
            "params": {"nprobe": 10}
        }
        
        results = self.collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            expr=filter_expr,
            output_fields=output_fields
        )
        
        # Format results
        formatted_results = []
        for hits in results:
            for hit in hits:
                result = {
                    "id": hit.id,
                    "score": hit.score,
                    "distance": hit.distance,
                }
                # Add output fields
                for field in output_fields:
                    result[field] = hit.entity.get(field)
                formatted_results.append(result)
        
        return formatted_results

    def hybrid_search(
        self, 
        query: str, 
        query_embedding: List[float], 
        top_k: int = 5, 
        filter_expr: Optional[str] = None, 
        output_fields: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Hybrid search for similar vectors.
        
        Args:
            query_embedding: Query vector
            top_k: Number of results to return
        """
        if output_fields is None:
            output_fields = ["text", "metadata"]
        
        search_params = {
            "metric_type": self.config.metric_type,
            "params": {"nprobe": 10}
        }
        
        req_dense = AnnSearchRequest(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
        )

        sparse_search_params = {"metric_type": "BM25"}
        req_sparse = AnnSearchRequest(
            data=[query],
            anns_field="sparse_vector",
            param=sparse_search_params,
            limit=top_k,
        )

        reqs = [req_dense, req_sparse]

        ranker = RRFRanker()

        results = self.collection.hybrid_search(
            reqs=reqs,
            rerank=ranker,
            output_fields=output_fields,
            limit=top_k
        )

        # Format results
        formatted_results = []
        for hits in results:
            for hit in hits:
                result = {
                    "id": hit.id,
                    "score": hit.score,
                    "distance": hit.distance,
                }
                # Add output fields
                for field in output_fields:
                    result[field] = hit.entity.get(field)
                formatted_results.append(result)
        
        return formatted_results
    
    def delete(self, expr: str) -> int:
        """
        Delete entities based on expression.
        
        Args:
            expr: Delete expression (e.g., 'id in [1, 2, 3]')
        
        Returns:
            Number of deleted entities
        """
        self.collection.delete(expr)
        self.collection.flush()
        logger.info(f"✅ Deleted entities matching: {expr}")
        return 0  # Milvus doesn't return count directly
    
    def count(self) -> int:
        """Get the number of entities in the collection."""
        return self.collection.num_entities
    
    def drop_collection(self):
        """Drop the entire collection. Use with caution!"""
        collection_name = self.config.collection_name
        self.collection.drop()
        logger.info(f"⚠️  Dropped collection: {collection_name}")
        self.collection = None
    
    def disconnect(self):
        """Disconnect from Milvus."""
        connections.disconnect(self.config.uri)
        logger.info(f"✅ Disconnected from Milvus")
    
    def __repr__(self) -> str:
        """String representation of the vector store."""
        return (
            f"VectorStore(collection='{self.config.collection_name}', "
            f"uri='{self.config.uri}', "
            f"token='{self.config.token}', "
            f"dimension={self.config.dimension}, "
            f"entities={self.count() if self.collection else 0})"
        )


if __name__ == "__main__":
    # Example usage
    from loader import DocumentLoader
    from chunker import Chunker
    from embedding import EmbeddingModel
    
    # Load and process documents
    loader = DocumentLoader()
    document = loader.load("./mini/documents/eb_test.pdf")
    
    # Chunk the document
    chunker = Chunker()
    chunks = chunker.chunk(document)
    
    # Generate embeddings
    embedding_model = EmbeddingModel()
    embeddings = embedding_model.embed_chunks(chunks)
    
    # Initialize vector store
    store = VectorStore(
        uri=os.getenv("MILVUS_URI"),
        token=os.getenv("MILVUS_TOKEN"),
        collection_name="eb_test",
        dimension=1536,
        metric_type="IP",
        index_type="IVF_FLAT",
        nlist=128,
    )
    
    # Insert embeddings
    texts = [chunk.text for chunk in chunks]
    metadata = [{"chunk_index": i, "source": "eb_test.pdf"} for i in range(len(chunks))]
    
    ids = store.insert(
        embeddings=embeddings,
        texts=texts,
        metadata=metadata
    )
    
    logger.debug(f"\nTotal entities in collection: {store.count()}")
    
    # Search for similar vectors
    query_embedding = embeddings[0]  # Use first embedding as query
    results = store.search(
        query_embedding=query_embedding,
        top_k=3
    )
    
    logger.debug("\nSearch results normal search:")
    for i, result in enumerate(results, 1):
        logger.debug(f"\n{i}. Score: {result['score']:.4f}")
        logger.debug(f"   Text: {result['text'][:100]}...")
        logger.debug(f"   Metadata: {result['metadata']}")
    
    results = store.hybrid_search(
        query="tell me about package 1",
        query_embedding=embedding_model.embed_query("tell me about package 1"),
        top_k=3
    )
    logger.debug("\nSearch results hybrid search:")
    for i, result in enumerate(results, 1):
        logger.debug(f"\n{i}. Score: {result['score']:.4f}")
        logger.debug(f"   Text: {result['text'][:100]}...")
        logger.debug(f"   Metadata: {result['metadata']}")
    # Clean up
    # store.drop_collection()  # Uncomment to drop collection
    store.disconnect()

