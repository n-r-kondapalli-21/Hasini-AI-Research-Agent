"""
Text splitter module for RAG Knowledge System.

Splits document text into overlapping chunks suitable for embedding and retrieval.
"""

import hashlib
import logging
from typing import List, Dict, Any

logger = logging.getLogger("hasini.rag.text_splitter")


class RAGTextSplitter:
    """
    Handles chunking of text documents with metadata preservation.
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        """
        Args:
            chunk_size: Target maximum characters per chunk.
            chunk_overlap: Overlap in characters between consecutive chunks.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            self._splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", ". ", " ", ""],
            )
            self._use_langchain = True
        except ImportError:
            self._use_langchain = False
            logger.info("langchain_text_splitters not found, using built-in chunker.")

    def split_text(self, text: str) -> List[str]:
        """Split string content into a list of chunk strings."""
        if not text or not text.strip():
            return []

        if self._use_langchain:
            return self._splitter.split_text(text)

        # Fallback character-level splitting with separators
        return self._fallback_split(text)

    def _fallback_split(self, text: str) -> List[str]:
        """Simple recursive character splitter fallback."""
        chunks = []
        start = 0
        text_len = len(text)

        while start < text_len:
            end = min(start + self.chunk_size, text_len)
            
            # If not at end, try finding a line break or sentence boundary
            if end < text_len:
                boundary = text.rfind("\n\n", start, end)
                if boundary == -1 or boundary < start + self.chunk_size // 2:
                    boundary = text.rfind("\n", start, end)
                if boundary == -1 or boundary < start + self.chunk_size // 2:
                    boundary = text.rfind(". ", start, end)
                
                if boundary != -1 and boundary > start:
                    end = boundary + 1

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            # Move start window with overlap
            start = end - self.chunk_overlap if end < text_len else text_len
            if start >= end:
                start = end

        return chunks

    def create_chunks_with_metadata(
        self,
        doc_data: Dict[str, Any],
        timestamp: str,
    ) -> List[Dict[str, Any]]:
        """
        Split a loaded document dictionary into chunk dicts with comprehensive metadata.

        Args:
            doc_data: Dict with 'content', 'source', 'filename', 'file_type'
            timestamp: ISO formatted timestamp string

        Returns:
            List of dicts containing 'id', 'text', and 'metadata'.
        """
        content = doc_data["content"]
        source = doc_data["source"]
        filename = doc_data["filename"]
        file_type = doc_data["file_type"]

        raw_chunks = self.split_text(content)
        total_chunks = len(raw_chunks)

        # Hash source path/URL to get a stable base ID
        source_hash = hashlib.md5(source.encode("utf-8")).hexdigest()[:10]

        chunk_objects = []
        for idx, chunk_text in enumerate(raw_chunks):
            chunk_id = f"doc_{source_hash}_chunk_{idx}"

            metadata = {
                "source": source,
                "filename": filename,
                "file_type": file_type,
                "chunk_id": chunk_id,
                "chunk_index": idx,
                "total_chunks": total_chunks,
                "timestamp": timestamp,
            }

            chunk_objects.append({
                "id": chunk_id,
                "text": chunk_text,
                "metadata": metadata,
            })

        logger.info(
            "Split '%s' into %d chunk(s) (avg len: %d chars).",
            filename,
            total_chunks,
            sum(len(c["text"]) for c in chunk_objects) // max(total_chunks, 1),
        )

        return chunk_objects
