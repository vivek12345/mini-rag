"""
Reranker module for re-ranking retrieved documents.
Supports multiple backends: Cohere API, local models, and LLM-based scoring.
"""

import os
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@dataclass
class RerankResult:
    """Result from a re-ranking operation."""
    text: str
    score: float
    index: int  # Original index in the input list
    metadata: Optional[Dict[str, Any]] = None


class BaseReranker(ABC):
    """Base class for all rerankers."""
    
    @abstractmethod
    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None
    ) -> List[RerankResult]:
        """
        Rerank documents based on relevance to query.
        
        Args:
            query: The search query
            documents: List of document texts to rerank
            top_k: Number of top results to return (None = all)
        
        Returns:
            List of RerankResult objects sorted by relevance
        """
        pass
    
    @abstractmethod
    def __repr__(self) -> str:
        """String representation of the reranker."""
        pass


class CohereReranker(BaseReranker):
    """
    Reranker using Cohere's Rerank API.
    
    Examples:
        # Initialize with API key
        reranker = CohereReranker(api_key="your-cohere-api-key")
        
        # Or use environment variable COHERE_API_KEY
        reranker = CohereReranker()
        
        # Rerank documents
        results = reranker.rerank(
            query="What is machine learning?",
            documents=["ML is...", "Deep learning...", "Neural networks..."],
            top_k=3
        )
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "rerank-english-v3.0",
        max_chunks_per_doc: Optional[int] = None
    ):
        """
        Initialize Cohere reranker.
        
        Args:
            api_key: Cohere API key (defaults to COHERE_API_KEY env var)
            model: Cohere rerank model to use
            max_chunks_per_doc: Maximum chunks per document for Cohere API
        """
        try:
            import cohere
        except ImportError:
            raise ImportError(
                "Cohere package not installed. Install it with: pip install cohere"
            )
        
        self.api_key = api_key or os.getenv("COHERE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Cohere API key is required. Set COHERE_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        self.model = model
        self.max_chunks_per_doc = max_chunks_per_doc
        self.client = cohere.Client(api_key=self.api_key)
        print(f"✅ Initialized CohereReranker with model: {model}")
    
    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None
    ) -> List[RerankResult]:
        """Rerank documents using Cohere's API."""
        if not documents:
            return []
        
        try:
            # Call Cohere rerank API
            kwargs = {
                "query": query,
                "documents": documents,
                "model": self.model,
            }
            
            if top_k is not None:
                kwargs["top_n"] = top_k
            
            if self.max_chunks_per_doc is not None:
                kwargs["max_chunks_per_doc"] = self.max_chunks_per_doc
            
            response = self.client.rerank(**kwargs)
            
            # Convert to RerankResult
            results = []
            for result in response.results:
                results.append(RerankResult(
                    text=documents[result.index],
                    score=result.relevance_score,
                    index=result.index
                ))
            
            return results
            
        except Exception as e:
            print(f"⚠️  Cohere reranking failed: {e}")
            # Fallback: return original order
            return [
                RerankResult(text=doc, score=1.0 - i/len(documents), index=i)
                for i, doc in enumerate(documents[:top_k] if top_k else documents)
            ]
    
    def __repr__(self) -> str:
        return f"CohereReranker(model='{self.model}')"


class SentenceTransformerReranker(BaseReranker):
    """
    Reranker using local sentence-transformers cross-encoder models.
    
    Examples:
        # Initialize with default model
        reranker = SentenceTransformerReranker()
        
        # Use a specific model
        reranker = SentenceTransformerReranker(
            model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
        )
        
        # Rerank documents
        results = reranker.rerank(
            query="What is machine learning?",
            documents=["ML is...", "Deep learning...", "Neural networks..."],
            top_k=3
        )
    """
    
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: Optional[str] = None
    ):
        """
        Initialize sentence-transformers cross-encoder reranker.
        
        Args:
            model_name: HuggingFace model name for cross-encoder
            device: Device to run the model on ('cuda', 'cpu', or None for auto)
        """
        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            raise ImportError(
                "sentence-transformers package not installed. "
                "Install it with: pip install sentence-transformers"
            )
        
        self.model_name = model_name
        self.device = device
        
        print(f"🔄 Loading cross-encoder model: {model_name}...")
        self.model = CrossEncoder(model_name, device=device)
        print(f"✅ Initialized SentenceTransformerReranker")
    
    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None
    ) -> List[RerankResult]:
        """Rerank documents using cross-encoder model."""
        if not documents:
            return []
        
        try:
            # Create query-document pairs
            pairs = [[query, doc] for doc in documents]
            
            # Get relevance scores
            scores = self.model.predict(pairs)
            
            # Create results with original indices
            results = [
                RerankResult(text=doc, score=float(score), index=i)
                for i, (doc, score) in enumerate(zip(documents, scores))
            ]
            
            # Sort by score (descending)
            results.sort(key=lambda x: x.score, reverse=True)
            
            # Return top_k if specified
            if top_k is not None:
                results = results[:top_k]
            
            return results
            
        except Exception as e:
            print(f"⚠️  Cross-encoder reranking failed: {e}")
            # Fallback: return original order
            return [
                RerankResult(text=doc, score=1.0 - i/len(documents), index=i)
                for i, doc in enumerate(documents[:top_k] if top_k else documents)
            ]
    
    def __repr__(self) -> str:
        return f"SentenceTransformerReranker(model='{self.model_name}')"


class LLMReranker(BaseReranker):
    """
    Reranker using LLM-based relevance scoring.
    Compatible with OpenAI and other OpenAI-compatible APIs.
    
    Examples:
        # Initialize with OpenAI
        from openai import OpenAI
        client = OpenAI(api_key="sk-...")
        reranker = LLMReranker(client=client, model="gpt-4o-mini")
        
        # Rerank documents
        results = reranker.rerank(
            query="What is machine learning?",
            documents=["ML is...", "Deep learning...", "Neural networks..."],
            top_k=3
        )
    """
    
    def __init__(
        self,
        client: Any,
        model: str = "gpt-4o-mini",
        temperature: float = 0.3,
        max_tokens: int = 10,
        truncate_length: int = 500
    ):
        """
        Initialize LLM-based reranker.
        
        Args:
            client: OpenAI client instance (or compatible)
            model: LLM model name
            temperature: Temperature for scoring
            max_tokens: Maximum tokens for response
            truncate_length: Maximum characters per document for scoring
        """
        self.client = client
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.truncate_length = truncate_length
        print(f"✅ Initialized LLMReranker with model: {model}")
    
    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None
    ) -> List[RerankResult]:
        """Rerank documents using LLM scoring."""
        if not documents:
            return []
        
        results = []
        
        # Score each document
        for i, doc in enumerate(documents[:min(len(documents), 20)]):  # Limit for efficiency
            prompt = f"""Rate the relevance of the following text to the query on a scale of 0-100.

Query: {query}

Text:
{doc[:self.truncate_length]}...

Respond with ONLY a number between 0 and 100, where:
- 0 = Completely irrelevant
- 50 = Somewhat relevant
- 100 = Highly relevant and directly answers the query

Score:"""
            
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a relevance scoring expert."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                
                score_text = response.choices[0].message.content.strip()
                # Extract numeric value
                score = float(''.join(filter(lambda x: x.isdigit() or x == '.', score_text)))
                normalized_score = score / 100.0  # Normalize to 0-1
                
                results.append(RerankResult(
                    text=doc,
                    score=normalized_score,
                    index=i
                ))
                
            except Exception as e:
                print(f"⚠️  LLM scoring failed for document {i}: {e}")
                # Use fallback score
                results.append(RerankResult(
                    text=doc,
                    score=0.5,
                    index=i
                ))
        
        # Sort by score (descending)
        results.sort(key=lambda x: x.score, reverse=True)
        
        # Return top_k if specified
        if top_k is not None:
            results = results[:top_k]
        
        return results
    
    def __repr__(self) -> str:
        return f"LLMReranker(model='{self.model}')"


class NoOpReranker(BaseReranker):
    """
    No-op reranker that returns documents in original order.
    Useful for disabling reranking while maintaining the interface.
    """
    
    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None
    ) -> List[RerankResult]:
        """Return documents in original order without reranking."""
        results = [
            RerankResult(text=doc, score=1.0 - i/len(documents), index=i)
            for i, doc in enumerate(documents)
        ]
        
        if top_k is not None:
            results = results[:top_k]
        
        return results
    
    def __repr__(self) -> str:
        return "NoOpReranker()"


# Factory function for easy reranker creation
def create_reranker(
    reranker_type: str = "cohere",
    **kwargs
) -> BaseReranker:
    """
    Factory function to create a reranker.
    
    Args:
        reranker_type: Type of reranker ('cohere', 'sentence-transformer', 'llm', 'none')
        **kwargs: Additional arguments for the specific reranker
    
    Returns:
        Initialized reranker instance
    
    Examples:
        # Cohere reranker
        reranker = create_reranker('cohere', api_key='...', model='rerank-english-v3.0')
        
        # Sentence transformer reranker
        reranker = create_reranker('sentence-transformer', model_name='cross-encoder/ms-marco-MiniLM-L-6-v2')
        
        # LLM reranker
        from openai import OpenAI
        client = OpenAI()
        reranker = create_reranker('llm', client=client, model='gpt-4o-mini')
        
        # No reranking
        reranker = create_reranker('none')
    """
    reranker_type = reranker_type.lower()
    
    if reranker_type == "cohere":
        return CohereReranker(**kwargs)
    elif reranker_type in ["sentence-transformer", "cross-encoder", "local"]:
        return SentenceTransformerReranker(**kwargs)
    elif reranker_type == "llm":
        return LLMReranker(**kwargs)
    elif reranker_type in ["none", "noop", "disabled"]:
        return NoOpReranker()
    else:
        raise ValueError(
            f"Unknown reranker type: {reranker_type}. "
            f"Choose from: 'cohere', 'sentence-transformer', 'llm', 'none'"
        )


# Example usage
if __name__ == "__main__":
    # Sample documents and query
    query = "What is machine learning?"
    documents = [
        "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
        "Python is a programming language widely used in data science.",
        "Deep learning uses neural networks with multiple layers to process complex patterns.",
        "The weather today is sunny with a high of 75 degrees.",
        "Neural networks are inspired by biological neurons in the human brain."
    ]
    
    print("=" * 80)
    print("Testing Different Rerankers")
    print("=" * 80)
    print(f"\nQuery: {query}\n")
    
    # Test Cohere reranker (if API key available)
    if os.getenv("COHERE_API_KEY"):
        print("\n--- Cohere Reranker ---")
        try:
            cohere_reranker = create_reranker("cohere")
            results = cohere_reranker.rerank(query, documents, top_k=3)
            for i, result in enumerate(results, 1):
                print(f"{i}. [Score: {result.score:.4f}] {result.text[:80]}...")
        except Exception as e:
            print(f"Cohere reranker failed: {e}")
    
    # Test Sentence Transformer reranker
    print("\n--- Sentence Transformer Reranker ---")
    try:
        st_reranker = create_reranker("sentence-transformer")
        results = st_reranker.rerank(query, documents, top_k=3)
        for i, result in enumerate(results, 1):
            print(f"{i}. [Score: {result.score:.4f}] {result.text[:80]}...")
    except Exception as e:
        print(f"Sentence transformer reranker failed: {e}")
    
    # Test LLM reranker (if OpenAI key available)
    if os.getenv("OPENAI_API_KEY"):
        print("\n--- LLM Reranker ---")
        try:
            from openai import OpenAI
            client = OpenAI()
            llm_reranker = create_reranker("llm", client=client, model="gpt-4o-mini")
            results = llm_reranker.rerank(query, documents, top_k=3)
            for i, result in enumerate(results, 1):
                print(f"{i}. [Score: {result.score:.4f}] {result.text[:80]}...")
        except Exception as e:
            print(f"LLM reranker failed: {e}")
    
    print("\n" + "=" * 80)

