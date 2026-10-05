"""
Hasini AI Research Agent - RAG Knowledge System

This module provides an independent, manually managed RAG (Retrieval-Augmented Generation)
Knowledge System backed by ChromaDB vector storage with hybrid search capabilities.

Key principles:
- Manually managed knowledge base (documents added/indexed explicitly by user).
- NOT conversational memory or user interaction history.
- Independent of interfaces (Voice, Telegram, CLI, MCP).
- Configurable relevance thresholds and metadata tracing.
- Hybrid search combining vector similarity, BM25 keyword search, and cross-encoder reranking.
"""

from .vector_store import RAGVectorStore
from .indexer import RAGIndexer
from .retriever import RAGRetriever
from .document_loader import RAGDocumentLoader
from .text_splitter import RAGTextSplitter
from .bm25_index import BM25Index
from .fusion import reciprocal_rank_fusion, filter_by_relevance
from .reranker import CrossEncoderReranker, get_reranker

__all__ = [
    "RAGVectorStore",
    "RAGIndexer",
    "RAGRetriever",
    "RAGDocumentLoader",
    "RAGTextSplitter",
    "BM25Index",
    "reciprocal_rank_fusion",
    "filter_by_relevance",
    "CrossEncoderReranker",
    "get_reranker",
]
