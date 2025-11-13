#!/usr/bin/env python3
"""
Test script to verify the package structure is correct before publishing.
Run this to ensure all imports work correctly.
"""

def test_imports():
    """Test that all main imports work."""
    print("Testing imports...")
    
    try:
        from mini import (
            DocumentLoader,
            Chunker,
            EmbeddingModel,
            VectorStore,
            AgenticRAG,
            EmbeddingConfig,
            MilvusConfig,
            LLMConfig,
            RetrievalConfig,
            RerankerConfig,
            LangfuseConfig,
            BaseReranker,
            RerankResult,
            CohereReranker,
            SentenceTransformerReranker,
            LLMReranker,
            create_reranker,
            __version__,
        )
        print("✅ All imports successful!")
        print(f"📦 Package version: {__version__}")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_basic_instantiation():
    """Test that basic classes can be instantiated."""
    print("\nTesting basic instantiation...")
    
    try:
        from mini import DocumentLoader, Chunker
        
        # Test DocumentLoader
        loader = DocumentLoader()
        print("✅ DocumentLoader instantiated")
        
        # Test Chunker
        chunker = Chunker()
        print("✅ Chunker instantiated")
        
        return True
    except Exception as e:
        print(f"❌ Instantiation failed: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 50)
    print("Package Structure Test")
    print("=" * 50)
    
    test1 = test_imports()
    test2 = test_basic_instantiation()
    
    print("\n" + "=" * 50)
    if test1 and test2:
        print("✅ All tests passed! Package is ready.")
        print("\nNext steps:")
        print("1. Update pyproject.toml with your details")
        print("2. Update LICENSE with your name")
        print("3. Run: python -m build")
        print("4. Run: python -m twine upload dist/*")
    else:
        print("❌ Some tests failed. Please fix the issues above.")
    print("=" * 50)

if __name__ == "__main__":
    main()

