"""
Agentic RAG (Retrieval-Augmented Generation) module.
Provides intelligent document retrieval with query rewriting, re-ranking, and agentic decision-making.
"""

import os
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from openai import OpenAI
from dotenv import load_dotenv

from mini.loader import DocumentLoader
from mini.chunker import Chunker
from mini.embedding import EmbeddingModel, EmbeddingConfig
from mini.store import VectorStore
from mini.reranker import BaseReranker, create_reranker, LLMReranker
from mini.observability import LangfuseConfig
from langfuse import observe

# Load environment variables
load_dotenv()


@dataclass
class LLMConfig:
    """Configuration for LLM settings."""
    model: str = "gpt-4"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.7
    timeout: float = 60.0
    max_retries: int = 3


@dataclass
class RetrievalConfig:
    """Configuration for retrieval settings."""
    top_k: int = 5
    rerank_top_k: int = 3
    use_query_rewriting: bool = True
    use_reranking: bool = True
    use_hybrid_search: bool = False


@dataclass
class RerankerConfig:
    """Configuration for reranker."""
    type: str = "llm"  # "llm", "cohere", "sentence-transformer", "none"
    custom_reranker: Optional[BaseReranker] = None
    kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ObservabilityConfig:
    """Configuration for observability/monitoring."""
    enabled: bool = False
    public_key: Optional[str] = None
    secret_key: Optional[str] = None
    host: Optional[str] = None


@dataclass
class RetrievalResult:
    """Result from a retrieval operation."""
    text: str
    score: float
    metadata: Dict[str, Any]
    rank: int
    reranked_score: Optional[float] = None


@dataclass
class RAGResponse:
    """Response from the RAG system."""
    answer: str
    original_query: str
    rewritten_queries: List[str]
    retrieved_chunks: List[RetrievalResult]
    context_used: str
    metadata: Dict[str, Any]


class AgenticRAG:
    """
    An intelligent RAG system that can:
    - Rewrite queries for better retrieval
    - Retrieve relevant chunks from vector store
    - Re-rank results for relevance
    - Generate contextual answers
    - Make agentic decisions about retrieval strategy
    
    Examples:
        # Simple usage with defaults
        rag = AgenticRAG(
            vector_store=store,
            embedding_model=embedding_model
        )
        
        # Custom LLM configuration
        rag = AgenticRAG(
            vector_store=store,
            embedding_model=embedding_model,
            llm_config=LLMConfig(model="gpt-4o-mini", temperature=0.5)
        )
        
        # Use Cohere reranker
        rag_cohere = AgenticRAG(
            vector_store=store,
            embedding_model=embedding_model,
            reranker_config=RerankerConfig(
                type="cohere",
                kwargs={"api_key": "your-cohere-key", "model": "rerank-english-v3.0"}
            )
        )
        
        # Full configuration
        rag_full = AgenticRAG(
            vector_store=store,
            embedding_model=embedding_model,
            llm_config=LLMConfig(model="gpt-4o-mini", temperature=0.5),
            retrieval_config=RetrievalConfig(top_k=10, rerank_top_k=5),
            reranker_config=RerankerConfig(type="cohere", kwargs={"api_key": "key"}),
            observability_config=ObservabilityConfig(enabled=True)
        )
        
        # Use custom reranker instance
        from reranker import CohereReranker
        custom_reranker = CohereReranker(api_key="your-key")
        rag_custom = AgenticRAG(
            vector_store=store,
            embedding_model=embedding_model,
            reranker_config=RerankerConfig(custom_reranker=custom_reranker)
        )
        
        # Query the system
        response = rag.query("What is the budget allocation for education?")
        print(response.answer)
        print(f"Used {len(response.retrieved_chunks)} chunks")
        
        # Index new documents
        rag.index_document("path/to/document.pdf")
    """
    
    def __init__(
        self,
        vector_store: VectorStore,
        embedding_model: EmbeddingModel,
        llm_config: Optional[LLMConfig] = None,
        retrieval_config: Optional[RetrievalConfig] = None,
        reranker_config: Optional[RerankerConfig] = None,
        observability_config: Optional[ObservabilityConfig] = None,
    ):
        """
        Initialize the Agentic RAG system.
        
        Args:
            vector_store: Initialized VectorStore instance
            embedding_model: Initialized EmbeddingModel instance
            llm_config: LLM configuration (uses defaults if None)
            retrieval_config: Retrieval configuration (uses defaults if None)
            reranker_config: Reranker configuration (uses defaults if None)
            observability_config: Observability configuration (uses defaults if None)
        """
        # Store core components
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        
        # Use defaults if configs not provided
        self.llm_config = llm_config or LLMConfig()
        self.retrieval_config = retrieval_config or RetrievalConfig()
        self.reranker_config = reranker_config or RerankerConfig()
        self.observability_config = observability_config or ObservabilityConfig()
        
        # Extract commonly used values for convenience
        self.use_query_rewriting = self.retrieval_config.use_query_rewriting
        self.use_reranking = self.retrieval_config.use_reranking
        self.temperature = self.llm_config.temperature
        self.top_k = self.retrieval_config.top_k
        self.rerank_top_k = self.retrieval_config.rerank_top_k
        self.use_hybrid_search = self.retrieval_config.use_hybrid_search
        
        # Initialize LLM client
        llm_kwargs = {
            "api_key": self.llm_config.api_key or os.getenv("OPENAI_API_KEY"),
            "timeout": self.llm_config.timeout,
            "max_retries": self.llm_config.max_retries,
        }
        if self.llm_config.base_url or os.getenv("OPENAI_BASE_URL"):
            llm_kwargs["base_url"] = self.llm_config.base_url or os.getenv("OPENAI_BASE_URL")
        
        self.llm_client = OpenAI(**llm_kwargs)
        self.llm_model = self.llm_config.model
        
        # Initialize reranker
        if self.reranker_config.custom_reranker:
            self.reranker = self.reranker_config.custom_reranker
        elif self.use_reranking:
            # Create reranker based on type
            if self.reranker_config.type.lower() == "llm":
                # Use LLM-based reranker with the same client
                self.reranker = LLMReranker(
                    client=self.llm_client,
                    model=self.llm_model,
                    **self.reranker_config.kwargs
                )
            else:
                # Use factory for other types
                self.reranker = create_reranker(
                    self.reranker_config.type,
                    **self.reranker_config.kwargs
                )
        else:
            self.reranker = None

        # Initialize observability if enabled
        if self.observability_config.enabled:
            self.observability = LangfuseConfig(
                enabled=self.observability_config.enabled,
                public_key=self.observability_config.public_key,
                secret_key=self.observability_config.secret_key,
                host=self.observability_config.host,
            )
        
        # Initialize document processing components
        self.loader = DocumentLoader()
        self.chunker = Chunker()
        
        print(f"✅ Initialized AgenticRAG with:")
        print(f"   - LLM Model: {self.llm_model}")
        print(f"   - Query Rewriting: {self.use_query_rewriting}")
        print(f"   - Re-ranking: {self.use_reranking}")
        if self.use_reranking and self.reranker:
            print(f"   - Reranker: {self.reranker}")
        print(f"   - Top-K Retrieval: {self.top_k}")
        print(f"   - Re-rank Top-K: {self.rerank_top_k}")
        print(f"   - Hybrid Search: {self.use_hybrid_search}")
    
    @observe
    def rewrite_query(self, query: str, num_variations: int = 2) -> List[str]:
        """
        Rewrite the query to generate multiple variations for better retrieval.
        
        Args:
            query: Original user query
            num_variations: Number of query variations to generate
        
        Returns:
            List of rewritten queries (including the original)
        """
        if not self.use_query_rewriting:
            return [query]
        
        prompt = f"""You are a query expansion expert. Given a user query, generate {num_variations} alternative 
phrasings that would help retrieve relevant information from a document database.

Original Query: {query}

Generate {num_variations} alternative queries that:
1. Use different keywords but maintain the same intent
2. Are more specific or include related concepts
3. Would help find relevant documents that might not match the original phrasing

Respond with ONLY the alternative queries, one per line, without numbering or additional text."""

        try:
            response = self.llm_client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": "You are a query expansion expert."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.8,
                max_tokens=200
            )
            
            rewritten = response.choices[0].message.content.strip().split("\n")
            rewritten = [q.strip() for q in rewritten if q.strip()]
            
            # Always include the original query
            all_queries = [query] + rewritten
            print(f"🔄 Generated {len(all_queries)} query variations")
            return all_queries
            
        except Exception as e:
            print(f"⚠️  Query rewriting failed: {e}. Using original query.")
            return [query]
    
    @observe
    def retrieve(
        self,
        queries: List[str],
        top_k: Optional[int] = None
    ) -> List[RetrievalResult]:
        """
        Retrieve relevant chunks for multiple queries.
        
        Args:
            queries: List of query strings
            top_k: Number of results to retrieve per query
        
        Returns:
            List of RetrievalResult objects, deduplicated
        """
        if top_k is None:
            top_k = self.top_k
        
        all_results = {}  # Use dict to deduplicate by text
        
        for query in queries:
            # Generate query embedding
            query_embedding = self.embedding_model.embed_query(query)
            
            if self.use_hybrid_search:
                search_results = self.vector_store.hybrid_search(
                    query=query,
                    query_embedding=query_embedding,
                    top_k=top_k
                )
            else:
                search_results = self.vector_store.search(
                    query_embedding=query_embedding,
                    top_k=top_k
                )
            
            # Convert to RetrievalResult
            for rank, result in enumerate(search_results, 1):
                text = result.get("text", "")
                if text not in all_results:
                    all_results[text] = RetrievalResult(
                        text=text,
                        score=result.get("score", 0.0),
                        metadata=result.get("metadata", {}),
                        rank=rank
                    )
        
        # Sort by score
        results = sorted(all_results.values(), key=lambda x: x.score, reverse=True)
        print(f"📚 Retrieved {len(results)} unique chunks")
        return results
    
    @observe
    def rerank(
        self,
        query: str,
        results: List[RetrievalResult],
        top_k: Optional[int] = None
    ) -> List[RetrievalResult]:
        """
        Re-rank retrieved results using the configured reranker.
        
        Args:
            query: Original query
            results: List of retrieval results
            top_k: Number of results to keep after re-ranking
        
        Returns:
            Re-ranked list of RetrievalResult objects
        """
        if not self.use_reranking or not results or not self.reranker:
            return results[:top_k or self.rerank_top_k]
        
        if top_k is None:
            top_k = self.rerank_top_k
        
        try:
            # Extract texts for reranking
            documents = [result.text for result in results]
            
            # Use reranker
            rerank_results = self.reranker.rerank(
                query=query,
                documents=documents,
                top_k=top_k
            )
            
            # Map back to RetrievalResult objects
            reranked = []
            for rr in rerank_results:
                # Find the original result by index
                original_result = results[rr.index]
                # Update with reranked score
                original_result.reranked_score = rr.score
                reranked.append(original_result)
            
            print(f"🎯 Re-ranked to top {len(reranked)} most relevant chunks")
            return reranked
            
        except Exception as e:
            print(f"⚠️  Re-ranking failed: {e}. Using original order.")
            return results[:top_k]
    @observe
    def generate_answer(
        self,
        query: str,
        context_chunks: List[RetrievalResult]
    ) -> str:
        """
        Generate an answer using the retrieved context.
        
        Args:
            query: User's original query
            context_chunks: List of relevant text chunks
        
        Returns:
            Generated answer string
        """
        # Build context
        context = "\n\n".join([
            f"[Source {i+1}]\n{chunk.text}"
            for i, chunk in enumerate(context_chunks)
        ])
        
        prompt = f"""You are a helpful assistant that answers questions based on the provided context.
Use the following pieces of context to answer the question. If the answer is not in the context, 
say "I cannot answer this question based on the provided context."

Context:
{context}

Question: {query}

Answer:"""
        
        try:
            response = self.llm_client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that provides accurate answers based on context."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=1000
            )
            
            answer = response.choices[0].message.content.strip()
            return answer
            
        except Exception as e:
            return f"Error generating answer: {e}"
    
    @observe(name="rag_query")
    def query(
        self,
        query: str,
        top_k: Optional[int] = None,
        rerank_top_k: Optional[int] = None,
        return_sources: bool = True
    ) -> RAGResponse:
        """
        Main query method that orchestrates the entire RAG pipeline.
        
        Args:
            query: User's question
            top_k: Number of chunks to retrieve (overrides default)
            rerank_top_k: Number of chunks after re-ranking (overrides default)
            return_sources: Whether to include source chunks in response
        
        Returns:
            RAGResponse object with answer and metadata
        """
        print(f"\n🤖 Processing query: '{query}'")
        
        # Step 1: Query Rewriting
        rewritten_queries = self.rewrite_query(query)
        
        # Step 2: Retrieval
        if top_k is None:
            top_k = self.top_k
        
        if rerank_top_k is None:
            rerank_top_k = self.rerank_top_k
        
        retrieved_chunks = self.retrieve(rewritten_queries, top_k)
        
        # Step 3: Re-ranking
        final_chunks = self.rerank(query, retrieved_chunks, rerank_top_k)
        
        # Step 4: Answer Generation
        answer = self.generate_answer(query, final_chunks)
        
        # Build context string
        context_used = "\n\n".join([chunk.text for chunk in final_chunks])
        
        print(f"✅ Generated answer using {len(final_chunks)} chunks\n")
        
        return RAGResponse(
            answer=answer,
            original_query=query,
            rewritten_queries=rewritten_queries,
            retrieved_chunks=final_chunks if return_sources else [],
            context_used=context_used,
            metadata={
                "num_queries": len(rewritten_queries),
                "num_retrieved": len(retrieved_chunks),
                "num_final": len(final_chunks),
                "query_rewriting_enabled": self.use_query_rewriting,
                "reranking_enabled": self.use_reranking,
            }
        )
    
    def index_document(
        self,
        document_path: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Load, chunk, embed, and index a document.
        
        Args:
            document_path: Path to the document
            metadata: Optional metadata to attach to chunks
        
        Returns:
            Number of chunks indexed
        """
        print(f"\n📄 Indexing document: {document_path}")
        
        # Load document
        document_text = self.loader.load(document_path)
        
        # Chunk document
        chunks = self.chunker.chunk(document_text)
        print(f"   Split into {len(chunks)} chunks")
        
        # Generate embeddings
        embeddings = self.embedding_model.embed_chunks(chunks)
        print(f"   Generated {len(embeddings)} embeddings")
        
        # Prepare metadata
        chunk_metadata = []
        for i in range(len(chunks)):
            meta = {
                "source": document_path,
                "chunk_index": i,
                **(metadata or {})
            }
            chunk_metadata.append(meta)
        
        # Insert into vector store
        texts = [chunk.text for chunk in chunks]
        self.vector_store.insert(
            embeddings=embeddings,
            texts=texts,
            metadata=chunk_metadata
        )
        
        print(f"✅ Successfully indexed {len(chunks)} chunks from {document_path}\n")
        return len(chunks)
    
    def index_documents(
        self,
        document_paths: List[str],
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Index multiple documents.
        
        Args:
            document_paths: List of document paths
            metadata: Optional metadata to attach to all chunks
        
        Returns:
            Total number of chunks indexed
        """
        total_chunks = 0
        for path in document_paths:
            total_chunks += self.index_document(path, metadata)
        
        print(f"🎉 Indexed {total_chunks} total chunks from {len(document_paths)} documents")
        return total_chunks
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the RAG system."""
        return {
            "total_chunks": self.vector_store.count(),
            "vector_store": str(self.vector_store),
            "embedding_model": str(self.embedding_model),
            "llm_model": self.llm_model,
            "query_rewriting": self.use_query_rewriting,
            "reranking": self.use_reranking,
            "top_k": self.top_k,
            "rerank_top_k": self.rerank_top_k,
        }
    
    def __repr__(self) -> str:
        """String representation of the RAG system."""
        return (
            f"AgenticRAG(model={self.llm_model}, "
            f"chunks={self.vector_store.count()}, "
            f"rewriting={self.use_query_rewriting}, "
            f"reranking={self.use_reranking})"
        )


# Example usage
if __name__ == "__main__":
    # Initialize components
    embedding_model = EmbeddingModel()
    
    vector_store = VectorStore(
        uri=os.getenv("MILVUS_URI"),
        token=os.getenv("MILVUS_TOKEN"),
        collection_name="eb_test",
        dimension=1536
    )
    
    # Initialize RAG system with configuration objects
    rag = AgenticRAG(
        vector_store=vector_store,
        embedding_model=embedding_model,
        llm_config=LLMConfig(model="gpt-4o-mini"),
        retrieval_config=RetrievalConfig(
            top_k=10,
            rerank_top_k=3,
            use_query_rewriting=True,
            use_reranking=True,
            use_hybrid_search=True
        ),
        reranker_config=RerankerConfig(
            type="cohere",
            kwargs={
                "model": "rerank-english-v3.0",
                "api_key": os.getenv("COHERE_API_KEY")
            }
        )
    )
    
    # Index a document (if needed)
    # rag.index_document("./mini/documents/eb_test.pdf")
    
    # Query the system
    queries = [
        "tell me about package 1",
        "what is outpatient for female in package 2"
    ]
    for query in queries:
        response = rag.query(query)
        print("=" * 80)
        print(f"Query: {response.original_query}")
        print("=" * 80)
        print(f"\nAnswer:\n{response.answer}")
        print("\n" + "=" * 80)
        print(f"Metadata:")
        for key, value in response.metadata.items():
            print(f"  {key}: {value}")
        print("=" * 80)
        
        # Show sources
        print(f"\nSources used ({len(response.retrieved_chunks)}):")
        for i, chunk in enumerate(response.retrieved_chunks, 1):
            print(f"\n{i}. [Score: {chunk.reranked_score or chunk.score:.4f}]")
            print(f"   {chunk.text[:200]}...")
            print(f"   Metadata: {chunk.metadata}")
        print("=" * 80)
    

