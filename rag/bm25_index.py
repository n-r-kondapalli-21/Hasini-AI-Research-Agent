"""
BM25 keyword search index module for RAG Knowledge System.

Provides persistent BM25 keyword-based retrieval to complement vector similarity search.
Maintains document-chunk alignment with ChromaDB for proper result fusion.
"""

import os
import json
import logging
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import defaultdict

from config import CHROMADB_PATH

logger = logging.getLogger("hasini.rag.bm25_index")


class BM25Index:
    """
    Persistent BM25 keyword search index for lexical retrieval.
    """

    def __init__(self, index_path: Optional[str] = None):
        """
        Initialize BM25 index with persistent storage.

        Args:
            index_path: Path to store BM25 index data. Defaults to knowledge_base/bm25_index/.
        """
        if index_path is None:
            # Default to knowledge_base/bm25_index/ alongside ChromaDB
            db_dir = Path(CHROMADB_PATH).parent
            self.index_path = db_dir / "bm25_index"
        else:
            self.index_path = Path(index_path).resolve()

        self.index_path.mkdir(parents=True, exist_ok=True)

        self.index_file = self.index_path / "bm25_index.pkl"
        self.metadata_file = self.index_path / "bm25_metadata.json"

        self.index: Optional[Dict[str, Any]] = None
        self.metadata: Dict[str, Dict[str, Any]] = {}

        self._load_index()

    def _load_index(self):
        """Load BM25 index and metadata from disk if they exist."""
        if self.index_file.exists() and self.metadata_file.exists():
            try:
                with open(self.index_file, "rb") as f:
                    self.index = pickle.load(f)
                with open(self.metadata_file, "r", encoding="utf-8") as f:
                    self.metadata = json.load(f)
                logger.info(
                    "Loaded BM25 index from '%s' (%d documents indexed).",
                    self.index_path,
                    len(self.metadata),
                )
            except Exception as exc:
                logger.warning("Failed to load BM25 index: %s. Creating new index.", exc)
                self.index = None
                self.metadata = {}
        else:
            self.index = None
            self.metadata = {}

    def _save_index(self):
        """Save BM25 index and metadata to disk."""
        try:
            with open(self.index_file, "wb") as f:
                pickle.dump(self.index, f)
            with open(self.metadata_file, "w", encoding="utf-8") as f:
                json.dump(self.metadata, f, indent=2, ensure_ascii=False)
            logger.debug("BM25 index saved to '%s'.", self.index_path)
        except Exception as exc:
            logger.error("Failed to save BM25 index: %s", exc)

    def _tokenize(self, text: str) -> List[str]:
        """
        Simple tokenization for BM25 indexing.

        Args:
            text: Input text string.

        Returns:
            List of lowercase tokens.
        """
        # Simple whitespace and punctuation tokenization
        import re
        tokens = re.findall(r'\b\w+\b', text.lower())
        return tokens

    def _build_bm25_index(self, corpus: List[str], k1: float = 1.5, b: float = 0.75):
        """
        Build BM25 index from corpus.

        Args:
            corpus: List of document texts.
            k1: BM25 k1 parameter (term saturation).
            b: BM25 b parameter (length normalization).
        """
        if not corpus:
            self.index = {}
            return

        # Calculate document lengths and average
        doc_lengths = [len(self._tokenize(doc)) for doc in corpus]
        avg_doc_length = sum(doc_lengths) / len(doc_lengths) if doc_lengths else 0

        # Build vocabulary and document frequencies
        vocab = set()
        doc_freqs = defaultdict(int)
        term_doc_lists = defaultdict(list)

        for doc_idx, doc in enumerate(corpus):
            tokens = self._tokenize(doc)
            unique_tokens = set(tokens)
            for token in unique_tokens:
                doc_freqs[token] += 1
                term_doc_lists[token].append(doc_idx)
            vocab.update(tokens)

        # Calculate IDF for each term
        N = len(corpus)
        idf = {}
        for term in vocab:
            idf[term] = max(0, 1.0 - (doc_freqs[term] - 0.5 + 0.5) / (doc_freqs[term] + 0.5))

        # Store index
        self.index = {
            "corpus": corpus,
            "doc_lengths": doc_lengths,
            "avg_doc_length": avg_doc_length,
            "vocab": vocab,
            "doc_freqs": doc_freqs,
            "term_doc_lists": term_doc_lists,
            "idf": idf,
            "k1": k1,
            "b": b,
            "N": N,
        }

    def add_document_chunks(self, chunks: List[Dict[str, Any]], source: str):
        """
        Add document chunks to BM25 index.

        Args:
            chunks: List of chunk dicts with 'id', 'text', and 'metadata'.
            source: Document source path or URL.
        """
        if not chunks:
            return

        # Extract texts and IDs
        texts = [chunk["text"] for chunk in chunks]
        chunk_ids = [chunk["id"] for chunk in chunks]

        # Remove existing chunks for this source
        self.remove_document(source)

        # Rebuild index with new chunks
        all_texts = []
        all_ids = []
        all_metadata = {}

        # Add existing documents
        for existing_source, meta in self.metadata.items():
            if existing_source != source:
                all_texts.extend(meta.get("texts", []))
                all_ids.extend(meta.get("chunk_ids", []))
                all_metadata[existing_source] = meta

        # Add new chunks
        all_texts.extend(texts)
        all_ids.extend(chunk_ids)
        all_metadata[source] = {
            "texts": texts,
            "chunk_ids": chunk_ids,
            "chunk_count": len(chunks),
        }

        # Rebuild BM25 index
        self._build_bm25_index(all_texts)
        self.metadata = all_metadata
        self._save_index()

        logger.info("Added %d chunk(s) to BM25 index for source '%s'.", len(chunks), source)

    def remove_document(self, source: str) -> int:
        """
        Remove document chunks from BM25 index.

        Args:
            source: Document source path or URL.

        Returns:
            Number of chunks removed.
        """
        if source not in self.metadata:
            return 0

        removed_count = self.metadata[source].get("chunk_count", 0)
        del self.metadata[source]

        # Rebuild index without removed document
        all_texts = []
        all_ids = []

        for meta in self.metadata.values():
            all_texts.extend(meta.get("texts", []))
            all_ids.extend(meta.get("chunk_ids", []))

        if all_texts:
            self._build_bm25_index(all_texts)
        else:
            self.index = {}

        self._save_index()
        logger.info("Removed %d chunk(s) from BM25 index for source '%s'.", removed_count, source)
        return removed_count

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search BM25 index for relevant chunks.

        Args:
            query: Search query string.
            top_k: Number of top results to return.

        Returns:
            List of dicts with 'chunk_id', 'score', and 'rank'.
        """
        if self.index is None or not self.index.get("corpus"):
            logger.debug("BM25 index is empty. Returning 0 results.")
            return []

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        corpus = self.index["corpus"]
        doc_lengths = self.index["doc_lengths"]
        avg_doc_length = self.index["avg_doc_length"]
        idf = self.index["idf"]
        k1 = self.index["k1"]
        b = self.index["b"]

        # Calculate BM25 scores for each document
        scores = []
        for doc_idx, doc in enumerate(corpus):
            doc_tokens = self._tokenize(doc)
            doc_length = doc_lengths[doc_idx]

            score = 0.0
            for token in query_tokens:
                if token in idf:
                    # Count term frequency in document
                    tf = doc_tokens.count(token)
                    # BM25 formula
                    numerator = tf * (k1 + 1)
                    denominator = tf + k1 * (1 - b + b * (doc_length / avg_doc_length))
                    score += idf[token] * (numerator / denominator)

            scores.append((doc_idx, score))

        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)

        # Get top-k results
        results = []
        for rank, (doc_idx, score) in enumerate(scores[:top_k], 1):
            # Find chunk_id by looking up in metadata
            chunk_id = self._find_chunk_id_by_index(doc_idx)
            if chunk_id:
                results.append({
                    "chunk_id": chunk_id,
                    "score": round(score, 4),
                    "rank": rank,
                })

        logger.debug("BM25 search: query='%s', %d results returned.", query, len(results))
        return results

    def _find_chunk_id_by_index(self, doc_idx: int) -> Optional[str]:
        """Find chunk_id corresponding to document index in corpus."""
        current_idx = 0
        for meta in self.metadata.values():
            chunk_count = meta.get("chunk_count", 0)
            if current_idx <= doc_idx < current_idx + chunk_count:
                chunk_ids = meta.get("chunk_ids", [])
                relative_idx = doc_idx - current_idx
                if relative_idx < len(chunk_ids):
                    return chunk_ids[relative_idx]
            current_idx += chunk_count
        return None

    def clear_all(self) -> int:
        """Clear all documents from BM25 index."""
        total_chunks = sum(meta.get("chunk_count", 0) for meta in self.metadata.values())
        self.index = None
        self.metadata = {}
        self._save_index()
        logger.info("Cleared BM25 index (%d chunk(s) removed).", total_chunks)
        return total_chunks

    def get_stats(self) -> Dict[str, Any]:
        """Return statistics about BM25 index."""
        total_chunks = sum(meta.get("chunk_count", 0) for meta in self.metadata.values())
        return {
            "total_chunks": total_chunks,
            "total_documents": len(self.metadata),
            "index_path": str(self.index_path),
            "index_loaded": self.index is not None,
        }
