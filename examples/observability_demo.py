"""
Demo script showing Langfuse observability integration with Mini RAG.
This example demonstrates how to enable and use observability features.
"""

import os
import sys

# Add parent directory to path to import mini modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from mini.rag import AgenticRAG, LLMConfig, RetrievalConfig, RerankerConfig, ObservabilityConfig
from mini.embedding import EmbeddingModel
from mini.store import VectorStore
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def main():
    """Main demo function."""
    print("=" * 80)
    print("Mini RAG - Langfuse Observability Demo")
    print("=" * 80)
    print()
    
    # Check for Langfuse credentials
    if not os.getenv("LANGFUSE_PUBLIC_KEY") or not os.getenv("LANGFUSE_SECRET_KEY"):
        print("⚠️  Warning: Langfuse credentials not found in environment variables.")
        print("   Set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY to enable tracing.")
        print("   The demo will continue without observability enabled.")
        print()
        enable_obs = False
    else:
        print("✅ Langfuse credentials found. Observability will be enabled.")
        print()
        enable_obs = True
    
    # Initialize components
    print("Initializing components...")
    embedding_model = EmbeddingModel()
    
    vector_store = VectorStore(
        uri=os.getenv("MILVUS_URI"),
        token=os.getenv("MILVUS_TOKEN"),
        collection_name="observability_demo",
        dimension=1536
    )
    
    # Initialize RAG system with observability
    print("\nInitializing RAG system with observability...")
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
        reranker_config=RerankerConfig(type="llm"),  # Using LLM reranker for this demo
        observability_config=ObservabilityConfig(enabled=enable_obs)
    )
    
    # Demo 1: Document Indexing with Tracing
    print("\n" + "=" * 80)
    print("Demo 1: Document Indexing with Tracing")
    print("=" * 80)
    print()
    
    document_path = "./mini/documents/eb_test.pdf"
    if os.path.exists(document_path):
        print(f"Indexing document: {document_path}")
        print("(Check your Langfuse dashboard for 'document_indexing' trace)")
        print()
        
        num_chunks = rag.index_document(
            document_path,
            metadata={"demo": "observability", "type": "test"},
            user_id="demo_user",
            session_id="demo_session_1"
        )
        
        print(f"✅ Indexed {num_chunks} chunks")
    else:
        print(f"⚠️  Document not found: {document_path}")
        print("   Skipping indexing demo...")
    
    # Demo 2: Query with Full Pipeline Tracing
    print("\n" + "=" * 80)
    print("Demo 2: Query with Full Pipeline Tracing")
    print("=" * 80)
    print()
    
    query = "What is the main topic of the document?"
    print(f"Query: '{query}'")
    print("(Check your Langfuse dashboard for 'rag_query' trace with sub-spans)")
    print()
    
    response = rag.query(
        query,
        user_id="demo_user",
        session_id="demo_session_1"
    )
    
    print("\n" + "-" * 80)
    print(f"Answer:\n{response.answer}")
    print("-" * 80)
    print(f"\nMetadata:")
    for key, value in response.metadata.items():
        print(f"  {key}: {value}")
    
    # Demo 3: Multiple Queries in Same Session
    print("\n" + "=" * 80)
    print("Demo 3: Multiple Queries in Same Session")
    print("=" * 80)
    print("(These will be grouped in Langfuse under session_id='demo_session_2')")
    print()
    
    queries = [
        "What is mentioned about the budget?",
        "Are there any statistics in the document?",
    ]
    
    for i, q in enumerate(queries, 1):
        print(f"\nQuery {i}: '{q}'")
        response = rag.query(
            q,
            user_id="demo_user",
            session_id="demo_session_2"
        )
        print(f"Answer: {response.answer[:200]}...")
    
    # Summary
    print("\n" + "=" * 80)
    print("Demo Complete!")
    print("=" * 80)
    print()
    
    if enable_obs:
        print("🎉 All operations have been traced to Langfuse!")
        print()
        print("What you can see in your Langfuse dashboard:")
        print("  1. Individual traces for each query and indexing operation")
        print("  2. Sub-spans showing query rewriting, retrieval, reranking, and generation")
        print("  3. Latency metrics for each step")
        print("  4. Input/output data for debugging")
        print("  5. Sessions grouped by session_id")
        print("  6. User activity grouped by user_id")
        print()
        print("Visit: https://cloud.langfuse.com (or your self-hosted instance)")
    else:
        print("ℹ️  Observability was not enabled. To enable:")
        print("  1. Sign up at https://cloud.langfuse.com")
        print("  2. Get your API keys from project settings")
        print("  3. Add them to your .env file:")
        print("     LANGFUSE_PUBLIC_KEY=pk-lf-...")
        print("     LANGFUSE_SECRET_KEY=sk-lf-...")
        print("  4. Run this demo again")
    
    print()


if __name__ == "__main__":
    main()

