"""
Retriever module for RAG Knowledge System.

Performs similarity queries, applies relevance score thresholds, logs retrieval operations,
and formats retrieved context for agent consumption.

Supports hybrid search with BM25 keyword retrieval and cross-encoder reranking.
"""

import logging
from typing import List, Dict, Any, Optional

from config import (
    RAG_ENABLED,
    RAG_TOP_K,
    RAG_BM25_ENABLED,
    RAG_VECTOR_TOP_K,
    RAG_BM25_TOP_K,
    RAG_RRF_K,
    RAG_RERANKER_ENABLED,
    RAG_RERANKER_TOP_K,
    RAG_FINAL_TOP_K,
    RAG_RERANKER_THRESHOLD,
)
from .vector_store import RAGVectorStore
from .bm25_index import BM25Index
from .fusion import reciprocal_rank_fusion, filter_by_relevance
from .reranker import get_reranker

logger = logging.getLogger("hasini.rag.retriever")


class RAGRetriever:
    """
    Manages knowledge retrieval and context formatting for the Hasini Agent.

    Supports hybrid search combining vector similarity, BM25 keyword search,
    RRF fusion, and cross-encoder reranking.
    """

    def __init__(
        self,
        vector_store: Optional[RAGVectorStore] = None,
        bm25_index: Optional[BM25Index] = None,
    ):
        """
        Initialize retriever with vector store and optional BM25 index.

        Args:
            vector_store: ChromaDB vector store instance.
            bm25_index: BM25 keyword index instance.
        """
        self.vector_store = vector_store or RAGVectorStore()
        self.bm25_index = bm25_index or BM25Index()
        self.reranker = None
        if RAG_RERANKER_ENABLED:
            try:
                self.reranker = get_reranker()
            except Exception as exc:
                logger.warning("Failed to initialize reranker: %s. Reranking disabled.", exc)

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        enabled: Optional[bool] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search knowledge base using hybrid retrieval (vector + BM25 + reranking).

        Args:
            query: User's search query string.
            top_k: Number of final results to return. Defaults to config RAG_FINAL_TOP_K.
            enabled: Override for RAG enabled status. Defaults to config RAG_ENABLED.

        Returns:
            List of chunk dicts retrieved.
        """
        is_enabled = RAG_ENABLED if enabled is None else enabled
        if not is_enabled:
            logger.debug("RAG retrieval is disabled in configuration. Skipping search.")
            return []

        if not query or not query.strip():
            logger.debug("Empty query string. Skipping RAG retrieval.")
            return []

        query_clean = query.strip()
        final_k = top_k if top_k is not None else RAG_FINAL_TOP_K

        logger.info(
            "Executing hybrid RAG search for query: '%s' (final_top_k=%d)",
            query_clean,
            final_k,
        )

        # Step 1: Vector search
        vector_top_k = RAG_VECTOR_TOP_K
        vector_results = self.vector_store.query(query=query_clean, top_k=vector_top_k)

        logger.info(
            "Vector search: %d result(s) retrieved (top_k=%d).",
            len(vector_results),
            vector_top_k,
        )

        # Step 2: BM25 search (if enabled)
        bm25_results = []
        if RAG_BM25_ENABLED:
            bm25_top_k = RAG_BM25_TOP_K
            bm25_results = self.bm25_index.search(query=query_clean, top_k=bm25_top_k)
            logger.info(
                "BM25 search: %d result(s) retrieved (top_k=%d).",
                len(bm25_results),
                bm25_top_k,
            )

        # Step 3: RRF fusion
        if vector_results and bm25_results:
            fused_candidates = reciprocal_rank_fusion(
                vector_results=vector_results,
                bm25_results=bm25_results,
                k=RAG_RRF_K,
            )
            logger.info("RRF fusion: %d candidate(s) generated.", len(fused_candidates))
        elif vector_results:
            # Only vector results available
            fused_candidates = vector_results
            logger.info("RRF fusion skipped (no BM25 results). Using vector results only.")
        else:
            # No results from either method
            logger.info("No results from vector or BM25 search.")
            return []

        # Step 4: Fetch full chunk data for BM25-only results
        if bm25_results:
            fused_candidates = self._fetch_full_chunk_data(fused_candidates)

        # Step 5: Reranking (if enabled)
        if RAG_RERANKER_ENABLED and self.reranker and fused_candidates:
            reranker_top_k = RAG_RERANKER_TOP_K
            reranked = self.reranker.rerank(
                query=query_clean,
                chunks=fused_candidates,
                top_k=reranker_top_k,
            )
            logger.info("Reranking: %d chunk(s) reranked (top_k=%d).", len(reranked), reranker_top_k)

            # Filter by reranker threshold
            if RAG_RERANKER_THRESHOLD > 0:
                reranked = filter_by_relevance(
                    chunks=reranked,
                    threshold=RAG_RERANKER_THRESHOLD,
                    score_key="rerank_score",
                )
                logger.info(
                    "Reranker threshold filtering: %d -> %d chunks (threshold=%.2f).",
                    len(fused_candidates),
                    len(reranked),
                    RAG_RERANKER_THRESHOLD,
                )

            final_chunks = reranked
        else:
            # No reranking, use fused candidates directly
            final_chunks = fused_candidates

        # Step 6: Apply final top-k limit
        if final_k > 0:
            final_chunks = final_chunks[:final_k]

        # Log final results
        retrieved_sources = list({c["metadata"].get("filename", "unknown") for c in final_chunks})
        logger.info(
            "Final RAG results: %d chunk(s) returned from source(s): %s",
            len(final_chunks),
            ", ".join(retrieved_sources) if retrieved_sources else "none",
        )

        return final_chunks

    def _fetch_full_chunk_data(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Fetch full chunk text and metadata for candidates that may only have chunk_id.

        Args:
            candidates: List of candidate chunks with at least 'id' field.

        Returns:
            List of chunks with full data from vector store.
        """
        # Collect all chunk IDs
        chunk_ids = [c["id"] for c in candidates]

        # Fetch from vector store
        try:
            results = self.vector_store.collection.get(
                ids=chunk_ids,
                include=["documents", "metadatas"],
            )

            if not results or not results.get("ids"):
                return candidates

            # Build mapping from chunk_id to full data
            id_to_data = {}
            for chunk_id, doc_text, meta in zip(
                results["ids"],
                results["documents"],
                results["metadatas"],
            ):
                id_to_data[chunk_id] = {
                    "text": doc_text,
                    "metadata": meta,
                }

            # Update candidates with full data
            for candidate in candidates:
                chunk_id = candidate["id"]
                if chunk_id in id_to_data:
                    candidate["text"] = id_to_data[chunk_id]["text"]
                    candidate["metadata"] = id_to_data[chunk_id]["metadata"]

            return candidates

        except Exception as exc:
            logger.warning("Failed to fetch full chunk data: %s", exc)
            return candidates

    def get_formatted_context(
        self,
        query: str,
        top_k: Optional[int] = None,
        enabled: Optional[bool] = None,
    ) -> Optional[str]:
        """
        Retrieve relevant chunks and format them into a structured context string.

        Args:
            query: User prompt or query string.
            top_k: Optional top_k override.
            enabled: Optional enabled status override.

        Returns:
            Formatted RAG context string if relevant knowledge found, else None.
        """
        chunks = self.retrieve(query=query, top_k=top_k, enabled=enabled)
        if not chunks:
            return None

        return self.format_context(chunks)

    @staticmethod
    def format_context(chunks: List[Dict[str, Any]]) -> str:
        """
        Format chunk dictionaries into a clean markdown block with source metadata.

        Args:
            chunks: List of retrieved chunk dicts.

        Returns:
            Formatted context string.
        """
        if not chunks:
            return ""

        context_lines = [
            "=== RAG KNOWLEDGE BASE CONTEXT ===",
            "The following relevant context was retrieved from your manually managed knowledge base.",
            "Use this information to answer the user's question accurately. If the context is partial or incomplete,",
            "you may also combine it with your general knowledge or available tools.",
            "",
        ]

        for idx, chunk in enumerate(chunks, 1):
            meta = chunk["metadata"]
            filename = meta.get("filename", "Unknown")
            source = meta.get("source", "Unknown")
            chunk_id = meta.get("chunk_id", f"chunk_{idx}")
            similarity = chunk.get("similarity", 0.0)
            timestamp = meta.get("timestamp", "N/A")
            text = chunk.get("text", "").strip()

            # Build score information
            scores = [f"Similarity: {similarity:.2f}"]
            if "rerank_score" in chunk:
                scores.append(f"Rerank: {chunk['rerank_score']:.2f}")
            if "rrf_score" in chunk:
                scores.append(f"RRF: {chunk['rrf_score']:.4f}")

            header = f"[{idx}] Source: {filename} (ID: {chunk_id} | {', '.join(scores)} | Date: {timestamp})"
            context_lines.append(header)
            context_lines.append(f"Full Path/URL: {source}")
            context_lines.append("Content:")
            context_lines.append(text)
            context_lines.append("-" * 40)

        context_lines.append("=== END RAG KNOWLEDGE BASE CONTEXT ===")
        return "\n".join(context_lines)
