"""
Cross-Encoder reranker module for RAG Knowledge System.

Downloads and loads a local Cross-Encoder model for reranking retrieved chunks.
Uses cross-encoder/ms-marco-MiniLM-L-6-v2 for high-quality relevance scoring.
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from config import RAG_EMBEDDING_MODEL_PATH

logger = logging.getLogger("hasini.rag.reranker")

# Resolve project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def get_resolved_reranker_path() -> Path:
    """
    Resolve the local reranker model path relative to project root.
    """
    # Store in models/embeddings/ alongside the embedding model
    reranker_path = Path(RAG_EMBEDDING_MODEL_PATH).parent / "cross-encoder-ms-marco-MiniLM-L-6-v2"
    if reranker_path.is_absolute():
        return reranker_path.resolve()
    return (PROJECT_ROOT / reranker_path).resolve()


class CrossEncoderReranker:
    """
    Cross-Encoder reranker for re-scoring query-chunk pairs.
    """

    DEFAULT_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(self, model_name: Optional[str] = None, model_path: Optional[str] = None):
        """
        Initialize cross-encoder reranker.

        Args:
            model_name: HuggingFace model identifier. Defaults to ms-marco-MiniLM-L-6-v2.
            model_path: Local path to store/load model. Defaults to models/embeddings/.
        """
        self.model_name = model_name or self.DEFAULT_MODEL_NAME
        self.model_path = Path(model_path) if model_path else get_resolved_reranker_path()
        self.model = None
        self._load_model()

    def _load_model(self):
        """
        Load or download cross-encoder model locally.

        1. Check if model exists locally.
        2. If missing, download from Hugging Face and save locally.
        3. Load model from local directory.
        """
        # Check if local model directory exists and contains files
        is_model_local = (
            self.model_path.exists()
            and self.model_path.is_dir()
            and any(self.model_path.iterdir())
        )

        if not is_model_local:
            logger.info("Local reranker model not found at '%s'.", self.model_path)
            logger.info("Downloading reranker model '%s'...", self.model_name)

            try:
                from sentence_transformers import CrossEncoder
                import sys
                from io import StringIO

                self.model_path.mkdir(parents=True, exist_ok=True)

                # Suppress progress output during download
                original_stderr = sys.stderr
                original_stdout = sys.stdout
                sys.stderr = StringIO()
                sys.stdout = StringIO()

                try:
                    # Download from HuggingFace
                    model = CrossEncoder(self.model_name)
                finally:
                    sys.stderr = original_stderr
                    sys.stdout = original_stdout

                # Save locally
                model.save(str(self.model_path))
                logger.info("Reranker model downloaded and saved to '%s'.", self.model_path)
            except Exception as exc:
                logger.error("Failed to download reranker model '%s': %s", self.model_name, exc)
                raise RuntimeError(
                    f"Failed to download reranker model '{self.model_name}': {exc}"
                ) from exc
        else:
            logger.info("Loading local reranker model from '%s'...", self.model_path)

        # Load model from local directory
        try:
            from sentence_transformers import CrossEncoder
            import sys
            from io import StringIO

            # Suppress progress output during loading
            original_stderr = sys.stderr
            original_stdout = sys.stdout
            sys.stderr = StringIO()
            sys.stdout = StringIO()

            try:
                self.model = CrossEncoder(str(self.model_path))
            finally:
                sys.stderr = original_stderr
                sys.stdout = original_stdout

            logger.info("Using local reranker model: '%s'.", self.model_path)
        except Exception as exc:
            logger.error("Failed to load local reranker model from '%s': %s", self.model_path, exc)
            raise RuntimeError(
                f"Could not load local reranker model from '{self.model_path}': {exc}"
            ) from exc

    def rerank(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Rerank chunks based on query-chunk relevance scores.

        Args:
            query: Search query string.
            chunks: List of chunk dicts with 'id', 'text', and 'metadata'.
            top_k: Number of top results to return. Returns all if None.

        Returns:
            List of chunk dicts sorted by rerank score, with added 'rerank_score' field.
        """
        if not chunks:
            logger.debug("No chunks to rerank.")
            return []

        if self.model is None:
            logger.warning("Reranker model not loaded. Returning original chunks.")
            return chunks

        # Prepare query-chunk pairs
        pairs = [[query, chunk["text"]] for chunk in chunks]

        try:
            # Predict relevance scores
            scores = self.model.predict(pairs, batch_size=32, show_progress_bar=False)

            # Add rerank scores to chunks
            for chunk, score in zip(chunks, scores):
                chunk["rerank_score"] = float(score)

            # Sort by rerank score descending
            reranked = sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)

            # Apply top-k limit
            if top_k is not None and top_k > 0:
                reranked = reranked[:top_k]

            logger.info(
                "Reranked %d chunk(s) for query '%s', returning top %d.",
                len(chunks),
                query,
                len(reranked),
            )

            return reranked

        except Exception as exc:
            logger.error("Failed to rerank chunks: %s", exc)
            # Return original chunks on error
            return chunks


_reranker_instance = None


def get_reranker() -> CrossEncoderReranker:
    """
    Get singleton reranker instance.

    Returns:
        CrossEncoderReranker instance.
    """
    global _reranker_instance
    if _reranker_instance is None:
        _reranker_instance = CrossEncoderReranker()
    return _reranker_instance
