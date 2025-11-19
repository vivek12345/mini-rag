"""
Example demonstrating hybrid search capabilities in the RAG system.
Shows how to use semantic + BM25 keyword search for improved retrieval.
"""

import os
import sys
from dotenv import load_dotenv

# Add parent directory to path to import mini modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from mini.embedding import EmbeddingModel
from mini.store import VectorStore
from mini.rag import AgenticRAG, LLMConfig, RetrievalConfig, RerankerConfig

# Load environment variables
load_dotenv()


def demo_hybrid_search():
    """Demo using hybrid search (semantic + BM25)."""
    print("\n" + "=" * 80)
    print("DEMO 1: Hybrid Search (Semantic + BM25)")
    print("=" * 80)
    
    # Initialize components
    embedding_model = EmbeddingModel()
    
    # Create a new collection for hybrid search demo
    # Note: Hybrid search requires sparse_vector field which is auto-created
    vector_store = VectorStore(
        uri=os.getenv("MILVUS_URI"),
        token=os.getenv("MILVUS_TOKEN"),
        collection_name="hybrid_search_demo",
        dimension=1536
    )
    
    # Initialize RAG with hybrid search enabled
    rag = AgenticRAG(
        vector_store=vector_store,
        embedding_model=embedding_model,
        llm_config=LLMConfig(model="gpt-4o-mini"),
        retrieval_config=RetrievalConfig(
            top_k=10,
            rerank_top_k=3,
            use_query_rewriting=True,
            use_reranking=True,
            use_hybrid_search=True  # Enable hybrid search
        ),
        reranker_config=RerankerConfig(type="llm")
    )
    
    # Index a document if collection is empty
    if vector_store.count() == 0:
        print("\n📄 Indexing sample document...")
        document_path = os.path.join(
            os.path.dirname(__file__), 
            "..", 
            "mini", 
            "documents", 
            "eb_test.pdf"
        )
        if os.path.exists(document_path):
            rag.index_document(document_path)
        else:
            print(f"⚠️  Document not found at {document_path}")
            print("   Please ensure you have documents indexed before running queries.")
            return
    
    # Query with hybrid search
    query = "What is the budget allocated for railways?"
    print(f"\n🔍 Query: {query}")
    print("\nUsing hybrid search (combines semantic similarity + BM25 keyword matching)...")
    
    response = rag.query(query)
    
    print(f"\n✅ Answer:\n{response.answer}")
    print(f"\n📊 Retrieved {len(response.retrieved_chunks)} chunks:")
    for i, chunk in enumerate(response.retrieved_chunks, 1):
        score = chunk.reranked_score or chunk.score
        print(f"\n{i}. [Score: {score:.4f}]")
        print(f"   {chunk.text[:150]}...")


def compare_semantic_vs_hybrid():
    """Compare semantic-only search vs hybrid search."""
    print("\n" + "=" * 80)
    print("DEMO 2: Semantic Search vs Hybrid Search Comparison")
    print("=" * 80)
    
    # Initialize shared components
    embedding_model = EmbeddingModel()
    
    # Create separate collections for comparison
    vector_store_semantic = VectorStore(
        uri=os.getenv("MILVUS_URI"),
        token=os.getenv("MILVUS_TOKEN"),
        collection_name="semantic_only_demo",
        dimension=1536
    )
    
    vector_store_hybrid = VectorStore(
        uri=os.getenv("MILVUS_URI"),
        token=os.getenv("MILVUS_TOKEN"),
        collection_name="hybrid_search_demo",
        dimension=1536
    )
    
    # Initialize RAG systems
    rag_semantic = AgenticRAG(
        vector_store=vector_store_semantic,
        embedding_model=embedding_model,
        llm_config=LLMConfig(model="gpt-4o-mini"),
        retrieval_config=RetrievalConfig(
            top_k=10,
            rerank_top_k=3,
            use_query_rewriting=False,  # Disable for fair comparison
            use_reranking=False,  # Disable reranking to see raw search results
            use_hybrid_search=False  # Semantic only
        )
    )
    
    rag_hybrid = AgenticRAG(
        vector_store=vector_store_hybrid,
        embedding_model=embedding_model,
        llm_config=LLMConfig(model="gpt-4o-mini"),
        retrieval_config=RetrievalConfig(
            top_k=10,
            rerank_top_k=3,
            use_query_rewriting=False,  # Disable for fair comparison
            use_reranking=False,  # Disable reranking to see raw search results
            use_hybrid_search=True  # Hybrid search
        )
    )
    
    # Index documents if needed
    document_path = os.path.join(
        os.path.dirname(__file__), 
        "..", 
        "mini", 
        "documents", 
        "eb_test.pdf"
    )
    
    if vector_store_semantic.count() == 0 and os.path.exists(document_path):
        print("\n📄 Indexing documents for semantic-only collection...")
        rag_semantic.index_document(document_path)
    
    if vector_store_hybrid.count() == 0 and os.path.exists(document_path):
        print("📄 Indexing documents for hybrid search collection...")
        rag_hybrid.index_document(document_path)
    
    # Test queries that benefit from hybrid search
    test_queries = [
        "budget allocation for railways",  # Contains specific keywords
        "G20 theme",  # Short query with specific terms
        "digital payments initiatives",  # Mix of conceptual and specific terms
    ]
    
    for query in test_queries:
        print(f"\n{'=' * 80}")
        print(f"Query: {query}")
        print("=" * 80)
        
        # Semantic search results
        print("\n📊 SEMANTIC SEARCH RESULTS:")
        print("-" * 80)
        try:
            semantic_response = rag_semantic.query(query)
            print(f"Retrieved {len(semantic_response.retrieved_chunks)} chunks")
            for i, chunk in enumerate(semantic_response.retrieved_chunks[:3], 1):
                print(f"\n{i}. [Score: {chunk.score:.4f}]")
                print(f"   {chunk.text[:120]}...")
        except Exception as e:
            print(f"⚠️  Error: {e}")
        
        # Hybrid search results
        print("\n🔀 HYBRID SEARCH RESULTS:")
        print("-" * 80)
        try:
            hybrid_response = rag_hybrid.query(query)
            print(f"Retrieved {len(hybrid_response.retrieved_chunks)} chunks")
            for i, chunk in enumerate(hybrid_response.retrieved_chunks[:3], 1):
                print(f"\n{i}. [Score: {chunk.score:.4f}]")
                print(f"   {chunk.text[:120]}...")
        except Exception as e:
            print(f"⚠️  Error: {e}")


def demo_hybrid_with_reranking():
    """Demo hybrid search combined with reranking."""
    print("\n" + "=" * 80)
    print("DEMO 3: Hybrid Search + Reranking")
    print("=" * 80)
    print("Combining hybrid search (semantic + BM25) with LLM-based reranking")
    
    # Initialize components
    embedding_model = EmbeddingModel()
    
    vector_store = VectorStore(
        uri=os.getenv("MILVUS_URI"),
        token=os.getenv("MILVUS_TOKEN"),
        collection_name="hybrid_search_demo",
        dimension=1536
    )
    
    # Initialize RAG with both hybrid search and reranking
    rag = AgenticRAG(
        vector_store=vector_store,
        embedding_model=embedding_model,
        llm_config=LLMConfig(model="gpt-4o-mini"),
        retrieval_config=RetrievalConfig(
            top_k=15,  # Retrieve more candidates
            rerank_top_k=5,  # Keep top 5 after reranking
            use_query_rewriting=True,
            use_reranking=True,
            use_hybrid_search=True  # Enable hybrid search
        ),
        reranker_config=RerankerConfig(type="llm")  # Use LLM reranking
    )
    
    # Query
    query = "What are the key initiatives announced for infrastructure development?"
    print(f"\n🔍 Query: {query}")
    
    response = rag.query(query)
    
    print(f"\n✅ Answer:\n{response.answer}")
    print(f"\n📊 Pipeline Summary:")
    print(f"   - Query variations: {len(response.rewritten_queries)}")
    print(f"   - Initial retrieval: {response.metadata['num_retrieved']} chunks")
    print(f"   - After reranking: {len(response.retrieved_chunks)} chunks")
    print(f"\n📚 Top {len(response.retrieved_chunks)} chunks used:")
    for i, chunk in enumerate(response.retrieved_chunks, 1):
        score = chunk.reranked_score or chunk.score
        print(f"\n{i}. [Reranked Score: {score:.4f}]")
        print(f"   {chunk.text[:150]}...")


def demo_when_to_use_hybrid():
    """Demonstrate when hybrid search is most beneficial."""
    print("\n" + "=" * 80)
    print("DEMO 4: When to Use Hybrid Search")
    print("=" * 80)
    
    embedding_model = EmbeddingModel()
    
    vector_store = VectorStore(
        uri=os.getenv("MILVUS_URI"),
        token=os.getenv("MILVUS_TOKEN"),
        collection_name="hybrid_search_demo",
        dimension=1536
    )
    
    rag_hybrid = AgenticRAG(
        vector_store=vector_store,
        embedding_model=embedding_model,
        llm_config=LLMConfig(model="gpt-4o-mini"),
        retrieval_config=RetrievalConfig(
            top_k=10,
            rerank_top_k=3,
            use_hybrid_search=True
        )
    )
    
    # Queries that benefit from hybrid search
    beneficial_queries = [
        {
            "query": "railways budget allocation",
            "reason": "Contains specific technical terms and exact keywords"
        },
        {
            "query": "G20 summit theme",
            "reason": "Short query with proper nouns and specific terms"
        },
        {
            "query": "digital payment UPI initiatives",
            "reason": "Mix of conceptual terms (digital payment) and specific acronyms (UPI)"
        },
    ]
    
    print("\n✅ Hybrid search is beneficial for queries with:")
    print("   - Specific technical terms or proper nouns")
    print("   - Exact keyword matches needed")
    print("   - Mix of conceptual and specific terms")
    print("   - Short queries with important keywords")
    
    for example in beneficial_queries:
        print(f"\n{'=' * 80}")
        print(f"Query: {example['query']}")
        print(f"Why: {example['reason']}")
        print("-" * 80)
        
        try:
            response = rag_hybrid.query(example['query'])
            print(f"\nAnswer: {response.answer[:200]}...")
            print(f"\nTop chunk: {response.retrieved_chunks[0].text[:150] if response.retrieved_chunks else 'N/A'}...")
        except Exception as e:
            print(f"⚠️  Error: {e}")


if __name__ == "__main__":
    print("\n🚀 Mini-RAG Hybrid Search Demonstrations\n")
    
    # Run demos
    demo_hybrid_search()
    compare_semantic_vs_hybrid()
    demo_hybrid_with_reranking()
    demo_when_to_use_hybrid()
    
    print("\n" + "=" * 80)
    print("✅ All hybrid search demos completed!")
    print("=" * 80)
    print("\n💡 Key Takeaways:")
    print("   - Hybrid search combines semantic similarity with BM25 keyword matching")
    print("   - Best for queries with specific terms, proper nouns, or exact keywords")
    print("   - Works seamlessly with reranking for optimal results")
    print("   - Automatically creates sparse vectors using Milvus BM25 function")
    print("\n")

