# Mini RAG 🚀

A lightweight, modular, and production-ready Retrieval-Augmented Generation (RAG) library built with Python. Install with `uv add mini-rag` and start building intelligent document search and question-answering systems in minutes. Mini RAG provides advanced features like query rewriting, re-ranking, and agentic decision-making—all with a simple, pythonic API.

## ✨ Features

- **🤖 Agentic RAG**: Intelligent query processing with automatic query rewriting and result re-ranking
- **📄 Multi-format Support**: Load documents from PDF, DOCX, images, and more using MarkItDown
- **✂️ Smart Chunking**: Advanced text chunking with Chonkie for optimal context preservation
- **🔮 Flexible Embeddings**: Support for OpenAI, Azure OpenAI, and any OpenAI-compatible API
- **💾 Vector Storage**: Powered by Milvus for high-performance similarity search
- **🎯 Query Optimization**: Automatic query rewriting for better retrieval results
- **📊 Multiple Re-ranking Options**: Choose from Cohere API, local cross-encoders, or LLM-based re-ranking
- **📈 Observability**: Built-in Langfuse integration for tracing and monitoring
- **🔧 Modular Design**: Use individual components or the complete RAG pipeline

## 💡 Library Usage at a Glance

Install Mini RAG and get started in seconds:

```bash
# Install the library
uv add mini-rag
```

```python
# Create your RAG application
import os
from mini import (
    AgenticRAG,
    EmbeddingModel,
    VectorStore
)

# Setup (one time)
embedding_model = EmbeddingModel()
vector_store = VectorStore(
    uri=os.getenv("MILVUS_URI"),
    token=os.getenv("MILVUS_TOKEN"),
    collection_name="my_knowledge_base",
    dimension=1536
)
rag = AgenticRAG(vector_store=vector_store, embedding_model=embedding_model)

# Use it
rag.index_document("my_document.pdf")  # Add documents
response = rag.query("What is the budget?")  # Ask questions
print(response.answer)
```

Mini RAG handles all the complexity: document loading, chunking, embedding, vector storage, query rewriting, retrieval, re-ranking, and answer generation—all with just a few lines of code.

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      AgenticRAG System                       │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ DocumentLoader│    │   Chunker    │    │EmbeddingModel│
│  (MarkItDown) │───▶│  (Chonkie)   │───▶│   (OpenAI)   │
└──────────────┘    └──────────────┘    └──────────────┘
                                                 │
                                                 ▼
                                        ┌──────────────┐
                                        │ VectorStore  │
                                        │   (Milvus)   │
                                        └──────────────┘
```

## 📦 Installation

### Prerequisites

- Python >= 3.11
- OpenAI API key (or compatible API)
- Milvus instance (local or cloud)

### Install as a Library (Recommended)

The easiest way to use Mini RAG is to install it as a library:

```bash
# Install from PyPI
uv add mini-rag
```

That's it! You can now import and use Mini RAG in your projects:

```python
from mini.rag import AgenticRAG, LLMConfig, RetrievalConfig
from mini.embedding import EmbeddingModel
from mini.store import VectorStore
```

### Install from Source (For Development)

If you want to contribute or modify the library:

#### Using UV (Recommended for Development)

```bash
# Clone the repository
git clone https://github.com/vivek12345/mini-rag.git
cd mini-rag

# Install dependencies using uv
uv sync
```

#### Using pip

```bash
# Clone the repository
git clone https://github.com/vivek12345/mini-rag.git
cd mini-rag

# Install in editable mode
pip install -e .
```

### Dependencies

The library automatically installs the following dependencies:

- `chonkie[hub,openai,viz]>=1.4.1` - Smart text chunking
- `cohere>=5.0.0` - Cohere API for re-ranking
- `markitdown[all]>=0.1.3` - Multi-format document loading
- `pydantic>=2.12.4` - Data validation
- `pymilvus>=2.5.0` - Vector database client
- `python-dotenv>=1.2.1` - Environment variable management
- `sentence-transformers>=2.2.0` - Local cross-encoder models for re-ranking
- `langfuse>=2.0.0` - Observability and tracing
- `openai>=1.0.0` - OpenAI API client

## 🚀 Quick Start

This guide shows you how to use Mini RAG as a library in your own projects. After installing with `pip install mini-rag`, follow these steps:

### Configuration-Based API

Mini RAG uses a clean, configuration-based API that organizes settings into logical groups:

- **`LLMConfig`**: Configure your language model (model name, API keys, temperature, etc.)
- **`RetrievalConfig`**: Control retrieval behavior (top-k, query rewriting, re-ranking)
- **`RerankerConfig`**: Choose and configure your re-ranking strategy
- **`ObservabilityConfig`**: Enable Langfuse tracing and monitoring

This approach provides:
- ✨ **Better organization**: Related settings grouped together
- 🔧 **Easier maintenance**: Change one config without affecting others
- 📖 **Clearer code**: Self-documenting configuration objects
- 🎯 **Type safety**: Validated with Pydantic dataclasses

### 1. Set up environment variables

Create a `.env` file in your project directory:

```bash
# OpenAI Configuration
OPENAI_API_KEY=sk-your-api-key-here
OPENAI_BASE_URL=https://api.openai.com/v1  # Optional, for custom endpoints
EMBEDDING_MODEL=text-embedding-3-small

# Milvus Configuration
MILVUS_URI=https://your-milvus-instance.com
MILVUS_TOKEN=your-milvus-token

# Optional: Cohere Configuration (for Cohere re-ranking)
COHERE_API_KEY=your-cohere-api-key

# Optional: Langfuse Configuration (for observability)
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com  # Optional, defaults to cloud
```

### 2. Basic Usage

```python
import os
from mini import (
    AgenticRAG, 
    LLMConfig, 
    RetrievalConfig,
    EmbeddingModel,
    VectorStore
)
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize components
embedding_model = EmbeddingModel()

vector_store = VectorStore(
    uri=os.getenv("MILVUS_URI"),
    token=os.getenv("MILVUS_TOKEN"),
    collection_name="my_documents",
    dimension=1536  # For text-embedding-3-small
)

# Initialize RAG system
rag = AgenticRAG(
    vector_store=vector_store,
    embedding_model=embedding_model,
    llm_config=LLMConfig(model="gpt-4o-mini"),
    retrieval_config=RetrievalConfig(
        top_k=10,
        rerank_top_k=3,
        use_query_rewriting=True,
        use_reranking=True
    )
)

# Index documents
rag.index_document("path/to/your/document.pdf")

# Query the system
response = rag.query("What is the main topic of the document?")

print(f"Answer: {response.answer}")
print(f"\nSources used: {len(response.retrieved_chunks)}")
print(f"Query variations: {response.rewritten_queries}")
```

### 3. Minimal Example (5 Lines!)

Once you have your environment set up, using Mini RAG is incredibly simple:

```python
import os
from mini import (
    AgenticRAG,
    EmbeddingModel,
    VectorStore
)

# Initialize (using environment variables from .env)
embedding_model = EmbeddingModel()
vector_store = VectorStore(
    uri=os.getenv("MILVUS_URI"),
    token=os.getenv("MILVUS_TOKEN"),
    collection_name="my_docs",
    dimension=1536
)
rag = AgenticRAG(vector_store=vector_store, embedding_model=embedding_model)

# Index a document
rag.index_document("path/to/document.pdf")

# Ask a question
response = rag.query("What is this document about?")
print(response.answer)
```

That's it! Mini RAG handles query rewriting, retrieval, re-ranking, and answer generation automatically.

### 4. Enabling Observability with Langfuse

Mini RAG includes built-in support for Langfuse observability, allowing you to track and analyze your RAG pipeline's performance:

```python
from mini import AgenticRAG, LLMConfig, RetrievalConfig, ObservabilityConfig

# Enable observability when initializing RAG
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
    observability_config=ObservabilityConfig(enabled=True)
)

# Query the system (observability is automatically tracked)
response = rag.query("What is the main topic?")

# Index documents with tracing
rag.index_document("path/to/document.pdf")
```

**What gets tracked:**
- 🔍 Query rewriting operations
- 📚 Document retrieval metrics
- 🎯 Re-ranking performance
- 💬 LLM generation calls
- 📄 Document indexing pipeline
- ⏱️ Latency for each step
- 🎭 Input/output data for debugging

**Setup Langfuse:**

1. Sign up for a free account at [Langfuse Cloud](https://cloud.langfuse.com) or self-host
2. Get your API keys from the project settings
3. Add them to your `.env` file (see step 1 above)
4. Enable observability with `enable_observability=True`

**Benefits:**
- Monitor RAG pipeline performance in real-time
- Debug query rewriting and retrieval issues
- Track LLM costs and token usage
- Analyze user sessions and behavior
- Export data for custom analytics

## 📚 Detailed Usage

Mini RAG is designed to be used as a library in your Python projects. You can use the complete RAG pipeline or individual components based on your needs.

### Using Individual Components

One of Mini RAG's strengths is its modularity. You can import and use individual components in your own projects:

```python
# Import only what you need
from mini.loader import DocumentLoader
from mini.chunker import Chunker
from mini.embedding import EmbeddingModel
from mini.store import VectorStore
from mini.reranker import CohereReranker, SentenceTransformerReranker
from mini.rag import AgenticRAG, LLMConfig, RetrievalConfig

# Mix and match components as needed
loader = DocumentLoader()
chunker = Chunker()
embedding_model = EmbeddingModel()

# Build your own pipeline
text = loader.load("document.pdf")
chunks = chunker.chunk(text)
embeddings = embedding_model.embed_chunks(chunks)
```

### Document Loading

The `DocumentLoader` class supports multiple file formats:

```python
from mini.loader import DocumentLoader

loader = DocumentLoader()

# Load a single document
text = loader.load("document.pdf")

# Load multiple documents
texts = loader.load_documents([
    "document1.pdf",
    "document2.docx",
    "image.png"
])

# Load all documents from a directory
texts = loader.load_documents_from_directory("./documents/")
```

**Supported formats:**
- PDF files (`.pdf`)
- Word documents (`.docx`, `.doc`)
- Images with OCR (`.png`, `.jpg`, `.jpeg`)
- Text files (`.txt`, `.md`)
- And more via MarkItDown

### Text Chunking

The `Chunker` class provides intelligent text splitting:

```python
from mini.chunker import Chunker

# Initialize chunker (default: markdown recipe)
chunker = Chunker(lang="en")

# Chunk text
chunks = chunker.chunk(text)

# Each chunk has text and metadata
for chunk in chunks:
    print(f"Text: {chunk.text[:100]}...")
    print(f"Token count: {chunk.token_count}")
```

### Embedding Generation

Generate embeddings using OpenAI-compatible APIs:

```python
from mini.embedding import EmbeddingModel

# Using OpenAI
embedding_model = EmbeddingModel(
    api_key="sk-...",
    model="text-embedding-3-small"
)

# Using Azure OpenAI
embedding_model = EmbeddingModel(
    api_key="your-azure-key",
    base_url="https://your-resource.openai.azure.com/openai/deployments/your-deployment",
    model="text-embedding-ada-002"
)

# Using a local model (e.g., llama.cpp)
embedding_model = EmbeddingModel(
    api_key="not-needed",
    base_url="http://localhost:8080/v1",
    model="text-embedding"
)

# Embed chunks
embeddings = embedding_model.embed_chunks(chunks)

# Embed a single query
query_embedding = embedding_model.embed_query("What is this about?")
```

### Vector Storage

Manage embeddings with Milvus:

```python
from mini.store import VectorStore

# Initialize vector store
store = VectorStore(
    uri="https://your-milvus-instance.com",
    token="your-token",
    collection_name="documents",
    dimension=1536,
    metric_type="IP"  # Inner Product (cosine similarity)
)

# Insert embeddings
ids = store.insert(
    embeddings=embeddings,
    texts=["Text 1", "Text 2"],
    metadata=[
        {"source": "doc1.pdf", "page": 1},
        {"source": "doc1.pdf", "page": 2}
    ]
)

# Search for similar vectors
results = store.search(
    query_embedding=query_embedding,
    top_k=5,
    filter_expr='metadata["source"] == "doc1.pdf"'  # Optional filter
)

# Get collection statistics
count = store.count()
print(f"Total documents: {count}")

# Clean up (use with caution!)
# store.drop_collection()
store.disconnect()
```

### Agentic RAG Pipeline

The complete RAG system with intelligent features:

```python
from mini.rag import AgenticRAG, LLMConfig, RetrievalConfig, RerankerConfig

# Initialize with custom settings
rag = AgenticRAG(
    vector_store=vector_store,
    embedding_model=embedding_model,
    llm_config=LLMConfig(
        model="gpt-4o-mini",
        api_key=None,  # Uses OPENAI_API_KEY env var
        base_url=None,  # Uses OPENAI_BASE_URL env var
        temperature=0.7,  # LLM temperature
        timeout=60.0,
        max_retries=3
    ),
    retrieval_config=RetrievalConfig(
        top_k=10,  # Retrieve 10 chunks initially
        rerank_top_k=3,  # Keep top 3 after re-ranking
        use_query_rewriting=True,  # Generate query variations
        use_reranking=True  # Re-rank results
    ),
    reranker_config=RerankerConfig(
        type="llm"  # Use LLM-based reranking (default)
    )
)

# Index a document
num_chunks = rag.index_document(
    document_path="document.pdf",
    metadata={"category": "research", "year": 2024}
)

# Index multiple documents
rag.index_documents([
    "doc1.pdf",
    "doc2.docx",
    "doc3.txt"
])

# Query the system
response = rag.query(
    query="What are the key findings?",
    top_k=10,  # Override default
    rerank_top_k=3,  # Override default
    return_sources=True  # Include source chunks
)

# Access response components
print(f"Answer: {response.answer}")
print(f"\nOriginal query: {response.original_query}")
print(f"Query variations: {response.rewritten_queries}")
print(f"\nMetadata: {response.metadata}")

# Show sources
for i, chunk in enumerate(response.retrieved_chunks, 1):
    print(f"\nSource {i}:")
    print(f"  Score: {chunk.reranked_score:.4f}")
    print(f"  Text: {chunk.text[:200]}...")
    print(f"  Metadata: {chunk.metadata}")

# Get system statistics
stats = rag.get_stats()
print(f"System stats: {stats}")
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `OPENAI_API_KEY` | OpenAI API key | - | Yes |
| `OPENAI_BASE_URL` | Custom API endpoint | `https://api.openai.com/v1` | No |
| `EMBEDDING_MODEL` | Embedding model name | `text-embedding-3-small` | No |
| `MILVUS_URI` | Milvus server URI | - | Yes |
| `MILVUS_TOKEN` | Milvus authentication token | - | Yes |
| `COHERE_API_KEY` | Cohere API key (for Cohere re-ranking) | - | No |
| `LANGFUSE_PUBLIC_KEY` | Langfuse public key (for observability) | - | No |
| `LANGFUSE_SECRET_KEY` | Langfuse secret key (for observability) | - | No |
| `LANGFUSE_HOST` | Langfuse host URL | `https://cloud.langfuse.com` | No |

### Advanced Configuration

#### RAG Configuration Examples

##### Simple Configuration (with defaults)
```python
from mini.rag import AgenticRAG

# Minimal setup - uses all defaults
rag = AgenticRAG(
    vector_store=vector_store,
    embedding_model=embedding_model
)
```

##### Custom LLM Configuration
```python
from mini.rag import AgenticRAG, LLMConfig

rag = AgenticRAG(
    vector_store=vector_store,
    embedding_model=embedding_model,
    llm_config=LLMConfig(
        model="gpt-4o-mini",
        api_key="sk-...",  # Optional, defaults to env var
        base_url="https://api.openai.com/v1",  # Optional
        temperature=0.5,
        timeout=120.0,
        max_retries=5
    )
)
```

##### Full Configuration Example
```python
import os
from mini.rag import (
    AgenticRAG, LLMConfig, RetrievalConfig, 
    RerankerConfig, ObservabilityConfig
)

rag = AgenticRAG(
    vector_store=vector_store,
    embedding_model=embedding_model,
    llm_config=LLMConfig(
        model="gpt-4o-mini",
        temperature=0.7
    ),
    retrieval_config=RetrievalConfig(
        top_k=10,
        rerank_top_k=5,
        use_query_rewriting=True,
        use_reranking=True
    ),
    reranker_config=RerankerConfig(
        type="cohere",
        kwargs={
            "api_key": os.getenv("COHERE_API_KEY"),
            "model": "rerank-english-v3.0"
        }
    ),
    observability_config=ObservabilityConfig(
        enabled=True,
        public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
        secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
        host="https://cloud.langfuse.com"
    )
)
```

#### Embedding Configuration

```python
from mini.embedding import EmbeddingConfig, EmbeddingModel

config = EmbeddingConfig(
    api_key="sk-...",
    base_url="https://api.openai.com/v1",
    model="text-embedding-3-small",
    dimensions=None,  # Use model default
    timeout=60.0,
    max_retries=3
)

embedding_model = EmbeddingModel(config=config)
```

#### Vector Store Configuration

```python
from mini.store import MilvusConfig, VectorStore

config = MilvusConfig(
    uri="https://your-instance.com",
    token="your-token",
    collection_name="documents",
    dimension=1536,
    metric_type="IP",  # IP, L2, or COSINE
    index_type="IVF_FLAT",  # IVF_FLAT, IVF_SQ8, HNSW
    nlist=128  # Number of cluster units
)

store = VectorStore(config=config)
```

## 🎯 Key Features Explained

### Query Rewriting

Automatically generates multiple query variations to improve retrieval:

```python
# Original: "What is the budget for education?"
# Generated variations:
# - "How much funding is allocated to education?"
# - "Education sector financial allocation"
```

This helps retrieve relevant documents that might not match the exact wording of the original query.

### Re-ranking

Mini RAG supports multiple re-ranking strategies to improve retrieval quality:

#### 1. LLM-Based Re-ranking (Default)

Uses your LLM to score and re-rank retrieved chunks:

```python
from mini.rag import AgenticRAG, LLMConfig, RetrievalConfig, RerankerConfig

rag = AgenticRAG(
    vector_store=vector_store,
    embedding_model=embedding_model,
    llm_config=LLMConfig(model="gpt-4o-mini"),
    retrieval_config=RetrievalConfig(use_reranking=True),
    reranker_config=RerankerConfig(type="llm")  # Default
)
```

#### 2. Cohere Re-rank API

Use Cohere's specialized re-ranking models for superior results:

```python
rag = AgenticRAG(
    vector_store=vector_store,
    embedding_model=embedding_model,
    retrieval_config=RetrievalConfig(use_reranking=True),
    reranker_config=RerankerConfig(
        type="cohere",
        kwargs={
            "api_key": "your-cohere-key",  # Or set COHERE_API_KEY env var
            "model": "rerank-english-v3.0"
        }
    )
)
```

#### 3. Local Cross-Encoder Models

Use open-source sentence-transformer models for privacy and cost efficiency:

```python
rag = AgenticRAG(
    vector_store=vector_store,
    embedding_model=embedding_model,
    retrieval_config=RetrievalConfig(use_reranking=True),
    reranker_config=RerankerConfig(
        type="sentence-transformer",
        kwargs={
            "model_name": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "device": "cuda"  # Optional: "cpu" or "cuda"
        }
    )
)
```

#### 4. Custom Re-ranker

Provide your own reranker instance:

```python
from mini.reranker import CohereReranker

custom_reranker = CohereReranker(
    api_key="your-key",
    model="rerank-multilingual-v3.0"
)

rag = AgenticRAG(
    vector_store=vector_store,
    embedding_model=embedding_model,
    reranker_config=RerankerConfig(custom_reranker=custom_reranker)
)
```

The re-ranking process ensures that the most contextually relevant information is prioritized for answer generation.

### Metadata Filtering

Filter search results by metadata:

```python
results = store.search(
    query_embedding=embedding,
    top_k=5,
    filter_expr='metadata["year"] == 2024 and metadata["category"] == "research"'
)
```

## 🔌 Integrating into Your Application

Mini RAG is designed to be easily integrated into existing Python applications:

### As a FastAPI/Flask Backend

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from mini.rag import AgenticRAG
from mini.embedding import EmbeddingModel
from mini.store import VectorStore
import os

app = FastAPI()

# Initialize once at startup
@app.on_event("startup")
async def startup_event():
    global rag
    embedding_model = EmbeddingModel()
    vector_store = VectorStore(
        uri=os.getenv("MILVUS_URI"),
        token=os.getenv("MILVUS_TOKEN"),
        collection_name="knowledge_base",
        dimension=1536
    )
    rag = AgenticRAG(vector_store=vector_store, embedding_model=embedding_model)

class Query(BaseModel):
    question: str

@app.post("/ask")
async def ask_question(query: Query):
    try:
        response = rag.query(query.question)
        return {
            "answer": response.answer,
            "sources": len(response.retrieved_chunks),
            "metadata": response.metadata
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

### As a Chatbot Component

```python
from mini.rag import AgenticRAG
from mini.embedding import EmbeddingModel
from mini.store import VectorStore

class DocumentChatbot:
    def __init__(self, milvus_uri: str, milvus_token: str):
        embedding_model = EmbeddingModel()
        vector_store = VectorStore(
            uri=milvus_uri,
            token=milvus_token,
            collection_name="chatbot_kb",
            dimension=1536
        )
        self.rag = AgenticRAG(
            vector_store=vector_store,
            embedding_model=embedding_model
        )
        self.conversation_history = []
    
    def add_documents(self, document_paths: list):
        """Add documents to the knowledge base."""
        return self.rag.index_documents(document_paths)
    
    def chat(self, user_message: str) -> str:
        """Chat with context from indexed documents."""
        self.conversation_history.append({"role": "user", "content": user_message})
        response = self.rag.query(user_message)
        self.conversation_history.append({"role": "assistant", "content": response.answer})
        return response.answer
    
    def get_history(self):
        """Get conversation history."""
        return self.conversation_history

# Usage
chatbot = DocumentChatbot(os.getenv("MILVUS_URI"), os.getenv("MILVUS_TOKEN"))
chatbot.add_documents(["faq.pdf", "manual.pdf"])
answer = chatbot.chat("How do I reset my password?")
```

### In a Data Processing Pipeline

```python
from mini.loader import DocumentLoader
from mini.chunker import Chunker
from mini.embedding import EmbeddingModel
import pandas as pd

class DocumentProcessor:
    def __init__(self):
        self.loader = DocumentLoader()
        self.chunker = Chunker()
        self.embedding_model = EmbeddingModel()
    
    def process_documents(self, file_paths: list) -> pd.DataFrame:
        """Process multiple documents and return a DataFrame."""
        results = []
        
        for path in file_paths:
            text = self.loader.load(path)
            chunks = self.chunker.chunk(text)
            embeddings = self.embedding_model.embed_chunks(chunks)
            
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                results.append({
                    'source': path,
                    'chunk_id': i,
                    'text': chunk.text,
                    'embedding': embedding,
                    'token_count': chunk.token_count
                })
        
        return pd.DataFrame(results)

# Usage
processor = DocumentProcessor()
df = processor.process_documents(["doc1.pdf", "doc2.pdf"])
print(f"Processed {len(df)} chunks")
```

## 🔍 Examples

### Example 1: Building a Document QA System

```python
import os
from mini.rag import AgenticRAG, LLMConfig
from mini.embedding import EmbeddingModel
from mini.store import VectorStore
from dotenv import load_dotenv

load_dotenv()

# Setup
embedding_model = EmbeddingModel()
vector_store = VectorStore(
    uri=os.getenv("MILVUS_URI"),
    token=os.getenv("MILVUS_TOKEN"),
    collection_name="company_docs",
    dimension=1536
)

rag = AgenticRAG(
    vector_store=vector_store,
    embedding_model=embedding_model,
    llm_config=LLMConfig(model="gpt-4o-mini")
)

# Index company documents
documents = [
    "./docs/employee_handbook.pdf",
    "./docs/policies.pdf",
    "./docs/benefits.pdf"
]

for doc in documents:
    rag.index_document(doc)

# Interactive Q&A
while True:
    query = input("\nAsk a question (or 'quit' to exit): ")
    if query.lower() == 'quit':
        break
    
    response = rag.query(query)
    print(f"\n{response.answer}")
```

### Example 2: Research Paper Analysis

```python
# Index research papers
papers = [
    "./papers/paper1.pdf",
    "./papers/paper2.pdf",
    "./papers/paper3.pdf"
]

for i, paper in enumerate(papers):
    rag.index_document(
        paper,
        metadata={"paper_id": i, "type": "research"}
    )

# Analyze findings
queries = [
    "What are the main findings across all papers?",
    "What methodologies were used?",
    "What are the limitations mentioned?"
]

for query in queries:
    response = rag.query(query)
    print(f"\nQuery: {query}")
    print(f"Answer: {response.answer}")
    print("=" * 80)
```

### Example 3: Custom Embedding Provider

```python
from mini.rag import AgenticRAG, LLMConfig

# Use a custom embedding provider (e.g., local model)
embedding_model = EmbeddingModel(
    api_key="not-needed",
    base_url="http://localhost:8080/v1",
    model="my-local-model",
    dimensions=768  # Custom dimension
)

vector_store = VectorStore(
    uri=os.getenv("MILVUS_URI"),
    token=os.getenv("MILVUS_TOKEN"),
    collection_name="local_embeddings",
    dimension=768  # Match embedding dimension
)

rag = AgenticRAG(
    vector_store=vector_store,
    embedding_model=embedding_model,
    llm_config=LLMConfig(model="gpt-4o-mini")
)
```

### Example 4: Comparing Re-ranking Strategies

```python
from mini.rag import AgenticRAG, LLMConfig, RerankerConfig
from mini.embedding import EmbeddingModel
from mini.store import VectorStore
import os

# Initialize shared components
embedding_model = EmbeddingModel()
vector_store = VectorStore(
    uri=os.getenv("MILVUS_URI"),
    token=os.getenv("MILVUS_TOKEN"),
    collection_name="documents",
    dimension=1536
)

query = "What are the main findings?"

# Test different rerankers
rerankers = [
    ("Cohere", "cohere", {"model": "rerank-english-v3.0"}),
    ("Local Cross-Encoder", "sentence-transformer", {"model_name": "cross-encoder/ms-marco-MiniLM-L-6-v2"}),
    ("LLM-based", "llm", {})
]

for name, reranker_type, kwargs in rerankers:
    print(f"\nTesting {name} reranker:")
    
    rag = AgenticRAG(
        vector_store=vector_store,
        embedding_model=embedding_model,
        llm_config=LLMConfig(model="gpt-4o-mini"),
        reranker_config=RerankerConfig(
            type=reranker_type,
            kwargs=kwargs
        )
    )
    
    response = rag.query(query)
    print(f"Answer: {response.answer[:200]}...")
    print(f"Chunks used: {len(response.retrieved_chunks)}")
```

## 🧪 Testing

Run the example scripts to test each component:

```bash
# Test document loading
uv run -m mini.loader

# Test chunking
uv run -m mini.chunker

# Test embeddings
uv run -m mini.embedding

# Test vector store
uv run -m mini.store

# Test re-rankers
uv run -m mini.reranker

# Test full RAG pipeline
uv run -m mini.rag

# Run comprehensive reranking demo
uv run examples/reranking_demo.py
```

## 📝 API Reference

### AgenticRAG

```python
class AgenticRAG:
    def __init__(
        self,
        vector_store: VectorStore,
        embedding_model: EmbeddingModel,
        llm_config: Optional[LLMConfig] = None,
        retrieval_config: Optional[RetrievalConfig] = None,
        reranker_config: Optional[RerankerConfig] = None,
        observability_config: Optional[ObservabilityConfig] = None
    )
    
    def query(
        self,
        query: str,
        top_k: Optional[int] = None,
        rerank_top_k: Optional[int] = None,
        return_sources: bool = True
    ) -> RAGResponse
    
    def index_document(
        self,
        document_path: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> int
    
    def index_documents(
        self,
        document_paths: List[str],
        metadata: Optional[Dict[str, Any]] = None
    ) -> int
    
    def get_stats(self) -> Dict[str, Any]

# Configuration Classes

from dataclasses import dataclass, field

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
```

### DocumentLoader

```python
class DocumentLoader:
    def load(self, document_path: str) -> str
    def load_documents(self, document_paths: List[str]) -> List[str]
    def load_documents_from_directory(self, directory_path: str) -> List[str]
```

### Chunker

```python
class Chunker:
    def __init__(self, lang: str = "en")
    def chunk(self, text: str) -> List[Chunk]
```

### EmbeddingModel

```python
class EmbeddingModel:
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        dimensions: Optional[int] = None,
        timeout: float = 60.0,
        max_retries: int = 3
    )
    
    def embed_chunks(self, chunks: List[str]) -> List[List[float]]
    def embed_query(self, query: str) -> List[float]
```

### VectorStore

```python
class VectorStore:
    def __init__(
        self,
        uri: str,
        token: str,
        collection_name: Optional[str] = None,
        dimension: Optional[int] = None,
        metric_type: str = "IP",
        index_type: str = "IVF_FLAT",
        nlist: int = 128
    )
    
    def insert(
        self,
        embeddings: List[List[float]],
        texts: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> List[int]
    
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filter_expr: Optional[str] = None,
        output_fields: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]
    
    def count(self) -> int
    def delete(self, expr: str) -> int
    def drop_collection(self)
    def disconnect(self)
```

### Reranker

```python
# Factory function
def create_reranker(
    reranker_type: str = "cohere",  # 'cohere', 'sentence-transformer', 'llm', 'none'
    **kwargs
) -> BaseReranker

# Base reranker interface
class BaseReranker:
    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None
    ) -> List[RerankResult]

# Cohere reranker
class CohereReranker(BaseReranker):
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "rerank-english-v3.0",
        max_chunks_per_doc: Optional[int] = None
    )

# Sentence transformer reranker
class SentenceTransformerReranker(BaseReranker):
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: Optional[str] = None
    )

# LLM-based reranker
class LLMReranker(BaseReranker):
    def __init__(
        self,
        client: Any,  # OpenAI client
        model: str = "gpt-4o-mini",
        temperature: float = 0.3,
        max_tokens: int = 10,
        truncate_length: int = 500
    )
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [Chonkie](https://github.com/chonkie-inc/chonkie) - For smart text chunking
- [MarkItDown](https://github.com/microsoft/markitdown) - For multi-format document loading
- [Milvus](https://milvus.io/) - For vector database capabilities
- [OpenAI](https://openai.com/) - For embeddings and LLM APIs

## 📧 Support

For issues, questions, or contributions, please open an issue on GitHub or contact the maintainers.

---

Made with ❤️ by Vivek Nayyar

