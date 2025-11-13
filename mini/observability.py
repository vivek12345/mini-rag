"""
Observability module for Langfuse integration.
Provides configurable tracing for RAG operations.
"""

import os
from typing import Optional, Any, Dict, Callable
from functools import wraps
from contextlib import contextmanager
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class LangfuseConfig:
    """Configuration for Langfuse observability."""
    
    def __init__(
        self,
        enabled: bool = False,
        public_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        host: Optional[str] = None,
    ):
        """
        Initialize Langfuse configuration.
        
        Args:
            enabled: Whether to enable Langfuse tracing
            public_key: Langfuse public key (defaults to LANGFUSE_PUBLIC_KEY env var)
            secret_key: Langfuse secret key (defaults to LANGFUSE_SECRET_KEY env var)
            host: Langfuse host URL (defaults to LANGFUSE_HOST env var or cloud)
        """
        self.enabled = enabled
        self.public_key = public_key or os.getenv("LANGFUSE_PUBLIC_KEY")
        self.secret_key = secret_key or os.getenv("LANGFUSE_SECRET_KEY")
        self.host = host or os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
        
        self._client = None
        
        if self.enabled:
            self._initialize()
    
    def _initialize(self):
        """Initialize Langfuse client."""
        if not self.public_key or not self.secret_key:
            print("⚠️  Langfuse credentials not found. Disabling observability.")
            print("   Set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY environment variables.")
            self.enabled = False
            return
        
        try:
            from langfuse import Langfuse
            from langfuse.openai import openai as langfuse_openai
            
            self._client = Langfuse(
                public_key=self.public_key,
                secret_key=self.secret_key,
                host=self.host,
            )
            
            print(f"✅ Langfuse observability enabled (host: {self.host})")
            
        except ImportError:
            print("⚠️  Langfuse package not installed. Disabling observability.")
            print("   Install it with: pip install langfuse")
            self.enabled = False
        except Exception as e:
            print(f"⚠️  Failed to initialize Langfuse: {e}")
            self.enabled = False
    
    @property
    def client(self):
        """Get Langfuse client."""
        return self._client if self.enabled else None
    
    def is_enabled(self) -> bool:
        """Check if Langfuse is enabled."""
        return self.enabled and self._client is not None
    
    def flush(self):
        """Flush any pending traces."""
        if self.is_enabled():
            try:
                self._client.flush()
            except Exception as e:
                print(f"⚠️  Failed to flush Langfuse traces: {e}")
    
    def shutdown(self):
        """Shutdown Langfuse client and flush traces."""
        if self.is_enabled():
            try:
                self._client.shutdown()
            except Exception as e:
                print(f"⚠️  Failed to shutdown Langfuse: {e}")