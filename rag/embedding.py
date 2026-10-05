"""
Local SentenceTransformer embedding function module for Hasini RAG Knowledge System.

Downloads and loads the embedding model locally inside project's `models/embeddings/` directory.
Integrates with ChromaDB vector store using a custom EmbeddingFunction interface wrapper.
"""

import os
import logging
from pathlib import Path
from typing import List

from chromadb import EmbeddingFunction, Documents, Embeddings

from config import (
    RAG_EMBEDDING_MODEL_NAME,
    RAG_EMBEDDING_MODEL_PATH,
)

logger = logging.getLogger("hasini.rag.embedding")

# Resolve project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def get_resolved_model_path() -> Path:
    """
    Resolve the local embedding model path relative to project root.
    """
    configured_path = Path(RAG_EMBEDDING_MODEL_PATH)
    if configured_path.is_absolute():
        return configured_path.resolve()
    return (PROJECT_ROOT / configured_path).resolve()


class LocalSentenceTransformerEmbeddingFunction(EmbeddingFunction):
    """
    Custom ChromaDB EmbeddingFunction wrapper around a local SentenceTransformer model.
    """

    def __init__(self, model, model_name: str = "all-MiniLM-L6-v2"):
        self.model = model
        self._model_name = model_name

    @classmethod
    def name(cls) -> str:
        """Return the embedding function name identifier for ChromaDB."""
        return f"sentence-transformers-{RAG_EMBEDDING_MODEL_NAME.replace('/', '_')}"

    def get_config(self) -> dict:
        """Return configuration dictionary for ChromaDB serialization."""
        return {"model_name": self._model_name}

    @classmethod
    def build_from_config(cls, config: dict):
        """Reconstruct embedding function from config dictionary."""
        return get_embedding_function()

    def __call__(self, input: Documents) -> Embeddings:
        """
        Generate vector embeddings for input document texts.

        Args:
            input: List of text strings to embed.

        Returns:
            List of float vector lists.
        """
        if not input:
            return []

        try:
            embeddings = self.model.encode(
                input,
                convert_to_numpy=True,
                show_progress_bar=False,
                batch_size=32,
            )
            return embeddings.tolist()
        except Exception as exc:
            logger.error("Failed to generate text embeddings: %s", exc)
            raise RuntimeError(f"Embedding generation error: {exc}") from exc


_embedding_function_instance = None


def get_embedding_function() -> EmbeddingFunction:
    """
    Load or download the local SentenceTransformer model and return ChromaDB embedding function.

    1. Check whether model exists in `models/embeddings/all-MiniLM-L6-v2/`.
    2. If missing, download from Hugging Face and save locally.
    3. Load model directly from local directory on subsequent runs.
    4. Raise clear RuntimeError on failure (no fallback).
    """
    global _embedding_function_instance
    if _embedding_function_instance is not None:
        return _embedding_function_instance

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        logger.error("sentence-transformers package is missing. Install via pip install sentence-transformers.")
        raise RuntimeError(
            "sentence-transformers is required for RAG local embeddings. "
            "Please run: pip install sentence-transformers"
        ) from exc

    model_path = get_resolved_model_path()

    # Check if local model directory exists and contains files
    is_model_local = (
        model_path.exists()
        and model_path.is_dir()
        and any(model_path.iterdir())
    )

    if not is_model_local:
        logger.info("Local embedding model not found at '%s'.", model_path)
        logger.info("Downloading embedding model '%s'...", RAG_EMBEDDING_MODEL_NAME)

        try:
            model_path.mkdir(parents=True, exist_ok=True)
            # Download from HuggingFace with progress disabled
            import sys
            from io import StringIO
            original_stderr = sys.stderr
            original_stdout = sys.stdout
            sys.stderr = StringIO()
            sys.stdout = StringIO()

            try:
                model = SentenceTransformer(RAG_EMBEDDING_MODEL_NAME)
            finally:
                sys.stderr = original_stderr
                sys.stdout = original_stdout

            # Save locally
            model.save(str(model_path))
            logger.info("Embedding model downloaded and saved to '%s'.", model_path)
        except Exception as exc:
            logger.error("Failed to download embedding model '%s': %s", RAG_EMBEDDING_MODEL_NAME, exc)
            raise RuntimeError(
                f"Failed to download embedding model '{RAG_EMBEDDING_MODEL_NAME}': {exc}"
            ) from exc
    else:
        logger.info("Loading local embedding model from '%s'...", model_path)

    # Load model from local directory
    try:
        # Suppress progress output during loading
        import sys
        from io import StringIO
        original_stderr = sys.stderr
        original_stdout = sys.stdout
        sys.stderr = StringIO()
        sys.stdout = StringIO()

        try:
            model = SentenceTransformer(str(model_path))
        finally:
            sys.stderr = original_stderr
            sys.stdout = original_stdout

        logger.info("Using local embedding model: '%s'.", model_path)
        _embedding_function_instance = LocalSentenceTransformerEmbeddingFunction(
            model=model,
            model_name=RAG_EMBEDDING_MODEL_NAME,
        )
        return _embedding_function_instance
    except Exception as exc:
        logger.error("Failed to load local embedding model from '%s': %s", model_path, exc)
        raise RuntimeError(
            f"Could not load local embedding model from '{model_path}': {exc}"
        ) from exc
