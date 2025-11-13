"""
Example demonstrating different reranking strategies in the RAG system.
Shows how to use Cohere API, local models, and LLM-based reranking.
"""

import os
import sys
from dotenv import load_dotenv

# Add parent directory to path to import mini modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from mini.embedding import EmbeddingModel
from mini.store import VectorStore
from mini.rag import AgenticRAG, LLMConfig, RetrievalConfig, RerankerConfig
from mini.reranker import create_reranker, CohereReranker, SentenceTransformerReranker

# Load environment variables
load_dotenv()


def demo_llm_reranking():
    """Demo using LLM-based reranking (default behavior)."""
    print("\n" + "=" * 80)
    print("DEMO 1: LLM-Based Reranking")
    print("=" * 80)
    
    # Initialize components
    embedding_model = EmbeddingModel()
    
    vector_store = VectorStore(
        uri=os.getenv("MILVUS_URI"),
        token=os.getenv("MILVUS_TOKEN"),
        collection_name="eb_test",
        dimension=1536
    )
    
    # Initialize RAG with LLM reranking
    rag = AgenticRAG(
        vector_store=vector_store,
        embedding_model=embedding_model,
        llm_config=LLMConfig(model="gpt-4o-mini"),
        retrieval_config=RetrievalConfig(
            top_k=10,
            rerank_top_k=3,
            use_query_rewriting=True,
            use_reranking=True
        ),
        reranker_config=RerankerConfig(type="llm")  # Use LLM for reranking
    )
    
    # Query the system
    response = rag.query("What is the budget allocated for railways?")
    
    print(f"\nQuery: {response.original_query}")
    print(f"\nAnswer:\n{response.answer}")
    print(f"\nSources used: {len(response.retrieved_chunks)}")
    for i, chunk in enumerate(response.retrieved_chunks, 1):
        print(f"{i}. [Score: {chunk.reranked_score or chunk.score:.4f}] {chunk.text[:100]}...")


def demo_cohere_reranking():
    """Demo using Cohere's Rerank API."""
    print("\n" + "=" * 80)
    print("DEMO 2: Cohere API Reranking")
    print("=" * 80)
    
    # Check if Cohere API key is available
    if not os.getenv("COHERE_API_KEY"):
        print("⚠️  COHERE_API_KEY not found. Skipping Cohere demo.")
        return
    
    # Initialize components
    embedding_model = EmbeddingModel()
    
    vector_store = VectorStore(
        uri=os.getenv("MILVUS_URI"),
        token=os.getenv("MILVUS_TOKEN"),
        collection_name="eb_test",
        dimension=1536
    )
    
    # Initialize RAG with Cohere reranking
    rag = AgenticRAG(
        vector_store=vector_store,
        embedding_model=embedding_model,
        llm_config=LLMConfig(model="gpt-4o-mini"),
        retrieval_config=RetrievalConfig(
            top_k=10,
            rerank_top_k=3,
            use_query_rewriting=True,
            use_reranking=True
        ),
        reranker_config=RerankerConfig(
            type="cohere",  # Use Cohere for reranking
            kwargs={"model": "rerank-english-v3.0"}
        )
    )
    
    # Query the system
    response = rag.query("What is the theme of G20?")
    
    print(f"\nQuery: {response.original_query}")
    print(f"\nAnswer:\n{response.answer}")
    print(f"\nSources used: {len(response.retrieved_chunks)}")
    for i, chunk in enumerate(response.retrieved_chunks, 1):
        print(f"{i}. [Score: {chunk.reranked_score or chunk.score:.4f}] {chunk.text[:100]}...")


def demo_sentence_transformer_reranking():
    """Demo using local sentence-transformer cross-encoder."""
    print("\n" + "=" * 80)
    print("DEMO 3: Sentence Transformer (Local) Reranking")
    print("=" * 80)
    
    try:
        # Initialize components
        embedding_model = EmbeddingModel()
        
        vector_store = VectorStore(
            uri=os.getenv("MILVUS_URI"),
            token=os.getenv("MILVUS_TOKEN"),
            collection_name="eb_test",
            dimension=1536
        )
        
        # Initialize RAG with sentence-transformer reranking
        rag = AgenticRAG(
            vector_store=vector_store,
            embedding_model=embedding_model,
            llm_config=LLMConfig(model="gpt-4o-mini"),
            retrieval_config=RetrievalConfig(
                top_k=10,
                rerank_top_k=3,
                use_query_rewriting=True,
                use_reranking=True
            ),
            reranker_config=RerankerConfig(
                type="sentence-transformer",  # Use local cross-encoder
                kwargs={"model_name": "cross-encoder/ms-marco-MiniLM-L-6-v2"}
            )
        )
        
        # Query the system
        response = rag.query("What initiatives were announced for digital payments?")
        
        print(f"\nQuery: {response.original_query}")
        print(f"\nAnswer:\n{response.answer}")
        print(f"\nSources used: {len(response.retrieved_chunks)}")
        for i, chunk in enumerate(response.retrieved_chunks, 1):
            print(f"{i}. [Score: {chunk.reranked_score or chunk.score:.4f}] {chunk.text[:100]}...")
    
    except ImportError:
        print("⚠️  sentence-transformers not installed. Skipping local model demo.")
        print("   Install with: pip install sentence-transformers")


def demo_custom_reranker():
    """Demo using a custom reranker instance."""
    print("\n" + "=" * 80)
    print("DEMO 4: Custom Reranker Instance")
    print("=" * 80)
    
    # Check if Cohere API key is available
    if not os.getenv("COHERE_API_KEY"):
        print("⚠️  COHERE_API_KEY not found. Skipping custom reranker demo.")
        return
    
    # Initialize components
    embedding_model = EmbeddingModel()
    
    vector_store = VectorStore(
        uri=os.getenv("MILVUS_URI"),
        token=os.getenv("MILVUS_TOKEN"),
        collection_name="eb_test",
        dimension=1536
    )
    
    # Create a custom reranker with specific settings
    custom_reranker = CohereReranker(
        api_key=os.getenv("COHERE_API_KEY"),
        model="rerank-english-v3.0",
        max_chunks_per_doc=10
    )
    
    # Initialize RAG with custom reranker
    rag = AgenticRAG(
        vector_store=vector_store,
        embedding_model=embedding_model,
        llm_config=LLMConfig(model="gpt-4o-mini"),
        retrieval_config=RetrievalConfig(
            top_k=10,
            rerank_top_k=3,
            use_query_rewriting=True,
            use_reranking=True
        ),
        reranker_config=RerankerConfig(custom_reranker=custom_reranker)  # Pass custom reranker instance
    )
    
    # Query the system
    response = rag.query("What are the tax changes mentioned?")
    
    print(f"\nQuery: {response.original_query}")
    print(f"\nAnswer:\n{response.answer}")
    print(f"\nSources used: {len(response.retrieved_chunks)}")
    for i, chunk in enumerate(response.retrieved_chunks, 1):
        print(f"{i}. [Score: {chunk.reranked_score or chunk.score:.4f}] {chunk.text[:100]}...")


def compare_rerankers():
    """Compare different reranking strategies on the same query."""
    print("\n" + "=" * 80)
    print("COMPARISON: Different Reranking Strategies")
    print("=" * 80)
    
    query = "What is the budget allocated for railways?"
    
    # Initialize common components
    embedding_model = EmbeddingModel()
    vector_store = VectorStore(
        uri=os.getenv("MILVUS_URI"),
        token=os.getenv("MILVUS_TOKEN"),
        collection_name="eb_test",
        dimension=1536
    )
    
    rerankers_to_test = [
        ("LLM-based", "llm", {}),
    ]
    
    if os.getenv("COHERE_API_KEY"):
        rerankers_to_test.append(("Cohere", "cohere", {"model": "rerank-english-v3.0"}))
    
    try:
        # Test if sentence-transformers is available
        import sentence_transformers
        rerankers_to_test.append((
            "Sentence Transformer",
            "sentence-transformer",
            {"model_name": "cross-encoder/ms-marco-MiniLM-L-6-v2"}
        ))
    except ImportError:
        pass
    
    results = {}
    
    for name, reranker_type, kwargs in rerankers_to_test:
        print(f"\n--- Testing {name} Reranker ---")
        
        try:
            rag = AgenticRAG(
                vector_store=vector_store,
                embedding_model=embedding_model,
                llm_config=LLMConfig(model="gpt-4o-mini"),
                retrieval_config=RetrievalConfig(
                    top_k=10,
                    rerank_top_k=3,
                    use_query_rewriting=False,  # Disable for fair comparison
                    use_reranking=True
                ),
                reranker_config=RerankerConfig(
                    type=reranker_type,
                    kwargs=kwargs
                )
            )
            
            response = rag.query(query)
            results[name] = response
            
            print(f"Top 3 chunks:")
            for i, chunk in enumerate(response.retrieved_chunks, 1):
                print(f"{i}. [Score: {chunk.reranked_score or chunk.score:.4f}] {chunk.text[:80]}...")
        
        except Exception as e:
            print(f"❌ Failed: {e}")
    
    # Print comparison summary
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)
    for name, response in results.items():
        print(f"\n{name}:")
        print(f"  Chunks used: {len(response.retrieved_chunks)}")
        if response.retrieved_chunks:
            avg_score = sum(c.reranked_score or c.score for c in response.retrieved_chunks) / len(response.retrieved_chunks)
            print(f"  Avg rerank score: {avg_score:.4f}")
        print(f"  Answer preview: {response.answer[:150]}...")


if __name__ == "__main__":
    print("\n🚀 Mini-RAG Reranking Demonstrations\n")
    
    # Run individual demos
    demo_llm_reranking()
    demo_cohere_reranking()
    demo_sentence_transformer_reranking()
    demo_custom_reranker()
    
    # Run comparison
    compare_rerankers()
    
    print("\n✅ All demos completed!\n")

