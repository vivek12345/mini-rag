"""
Simple example showing how to use different rerankers with the RAG system.
"""

import os
from dotenv import load_dotenv

# Add parent directory to path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from mini.embedding import EmbeddingModel
from mini.store import VectorStore
from mini.rag import AgenticRAG, LLMConfig, RetrievalConfig, RerankerConfig

# Load environment variables
load_dotenv()


def main():
    """Main example function."""
    
    # Initialize shared components
    embedding_model = EmbeddingModel()
    
    vector_store = VectorStore(
        uri=os.getenv("MILVUS_URI"),
        token=os.getenv("MILVUS_TOKEN"),
        collection_name="eb_test",
        dimension=1536
    )
    
    # Example 1: Using default LLM-based reranking
    print("\n" + "=" * 80)
    print("Example 1: LLM-Based Reranking (Default)")
    print("=" * 80)
    
    rag_llm = AgenticRAG(
        vector_store=vector_store,
        embedding_model=embedding_model,
        llm_config=LLMConfig(model="gpt-4o-mini"),
        reranker_config=RerankerConfig(type="llm")  # This is the default
    )
    
    response = rag_llm.query("What is the budget for railways?")
    print(f"\nAnswer: {response.answer[:200]}...")
    
    
    # Example 2: Using Cohere reranking (if API key available)
    if os.getenv("COHERE_API_KEY"):
        print("\n" + "=" * 80)
        print("Example 2: Cohere Reranking")
        print("=" * 80)
        
        rag_cohere = AgenticRAG(
            vector_store=vector_store,
            embedding_model=embedding_model,
            llm_config=LLMConfig(model="gpt-4o-mini"),
            reranker_config=RerankerConfig(
                type="cohere",
                kwargs={"model": "rerank-english-v3.0"}
            )
        )
        
        response = rag_cohere.query("What is the budget for railways?")
        print(f"\nAnswer: {response.answer[:200]}...")
    else:
        print("\n⚠️  Skipping Cohere example - set COHERE_API_KEY to enable")
    
    
    # Example 3: Using local sentence-transformer reranking
    print("\n" + "=" * 80)
    print("Example 3: Local Sentence Transformer Reranking")
    print("=" * 80)
    print("Note: First run will download the model (~80MB)")
    
    try:
        rag_local = AgenticRAG(
            vector_store=vector_store,
            embedding_model=embedding_model,
            llm_config=LLMConfig(model="gpt-4o-mini"),
            reranker_config=RerankerConfig(
                type="sentence-transformer",
                kwargs={"model_name": "cross-encoder/ms-marco-MiniLM-L-6-v2"}
            )
        )
        
        response = rag_local.query("What is the budget for railways?")
        print(f"\nAnswer: {response.answer[:200]}...")
    except ImportError:
        print("⚠️  sentence-transformers not installed. Install with:")
        print("   pip install sentence-transformers")
    
    
    # Example 4: Using a custom reranker instance
    if os.getenv("COHERE_API_KEY"):
        print("\n" + "=" * 80)
        print("Example 4: Custom Reranker Instance")
        print("=" * 80)
        
        from mini.reranker import CohereReranker
        
        # Create a custom reranker with specific settings
        custom_reranker = CohereReranker(
            api_key=os.getenv("COHERE_API_KEY"),
            model="rerank-english-v3.0",
            max_chunks_per_doc=10
        )
        
        rag_custom = AgenticRAG(
            vector_store=vector_store,
            embedding_model=embedding_model,
            llm_config=LLMConfig(model="gpt-4o-mini"),
            reranker_config=RerankerConfig(custom_reranker=custom_reranker)  # Pass custom instance
        )
        
        response = rag_custom.query("What is the budget for railways?")
        print(f"\nAnswer: {response.answer[:200]}...")
    
    
    # Example 5: Disable reranking
    print("\n" + "=" * 80)
    print("Example 5: Without Reranking")
    print("=" * 80)
    
    rag_no_rerank = AgenticRAG(
        vector_store=vector_store,
        embedding_model=embedding_model,
        llm_config=LLMConfig(model="gpt-4o-mini"),
        retrieval_config=RetrievalConfig(use_reranking=False)
    )
    
    response = rag_no_rerank.query("What is the budget for railways?")
    print(f"\nAnswer: {response.answer[:200]}...")
    
    
    print("\n" + "=" * 80)
    print("✅ Examples completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()

