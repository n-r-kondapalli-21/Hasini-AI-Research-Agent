"""
Indexer module for RAG Knowledge System.

Provides high-level document lifecycle management: indexing, re-indexing, removal, and listing.
Maintains both ChromaDB vector store and BM25 keyword index for hybrid search.
"""

import os
from datetime import datetime, timezone
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from .document_loader import RAGDocumentLoader
from .text_splitter import RAGTextSplitter
from .vector_store import RAGVectorStore
from .bm25_index import BM25Index

logger = logging.getLogger("hasini.rag.indexer")


class RAGIndexer:
    """
    Manages indexing lifecycle of user documents into ChromaDB vector database
    and BM25 keyword index for hybrid search.
    """

    def __init__(
        self,
        vector_store: Optional[RAGVectorStore] = None,
        bm25_index: Optional[BM25Index] = None,
        chunk_size: int = 600,
        chunk_overlap: int = 100,
    ):
        self.vector_store = vector_store or RAGVectorStore()
        self.bm25_index = bm25_index or BM25Index()
        self.text_splitter = RAGTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def index_document(self, source: str) -> Dict[str, Any]:
        """
        Load, split, and index a single document or URL into the knowledge base.

        Args:
            source: File path (e.g., 'path/to/doc.pdf') or web URL.

        Returns:
            Dict containing indexing result status, source, filename, and chunk count.
        """
        logger.info("Indexing document source: %s", source)

        try:
            # 1. Load document
            doc_data = RAGDocumentLoader.load(source)

            # 2. Generate ISO timestamp
            timestamp = datetime.now(timezone.utc).isoformat()

            # 3. Split into chunks with metadata
            chunks = self.text_splitter.create_chunks_with_metadata(
                doc_data=doc_data,
                timestamp=timestamp,
            )

            if not chunks:
                logger.warning("No chunks generated for document: %s", source)
                return {
                    "success": False,
                    "source": source,
                    "filename": doc_data["filename"],
                    "chunks_indexed": 0,
                    "message": "Document contains no extractable text chunks.",
                }

            # 4. Remove previous version if re-indexing existing source
            self.vector_store.delete_document(doc_data["source"])
            self.bm25_index.remove_document(doc_data["source"])

            # 5. Store chunks in ChromaDB
            added_count = self.vector_store.add_chunks(chunks)

            # 6. Add chunks to BM25 index
            self.bm25_index.add_document_chunks(chunks, doc_data["source"])

            logger.info(
                "Successfully indexed '%s' (%d chunks added to vector store and BM25 index).",
                doc_data["filename"],
                added_count,
            )

            return {
                "success": True,
                "source": doc_data["source"],
                "filename": doc_data["filename"],
                "file_type": doc_data["file_type"],
                "chunks_indexed": added_count,
                "timestamp": timestamp,
                "message": f"Successfully indexed {added_count} chunk(s).",
            }

        except Exception as exc:
            logger.error("Failed to index document '%s': %s", source, exc)
            return {
                "success": False,
                "source": source,
                "filename": os.path.basename(source),
                "chunks_indexed": 0,
                "message": f"Indexing error: {exc}",
            }

    def remove_document(self, source: str) -> Dict[str, Any]:
        """
        Remove an indexed document or URL from both vector store and BM25 index.

        Args:
            source: File path or URL to remove.

        Returns:
            Dict with deletion status and count of chunks removed.
        """
        logger.info("Removing document source: %s", source)

        # Resolve path if local file
        resolved_source = source
        if not (source.startswith("http://") or source.startswith("https://")):
            try:
                resolved_source = str(Path(source).resolve())
            except Exception:
                pass

        removed_count = self.vector_store.delete_document(resolved_source)
        bm25_removed = self.bm25_index.remove_document(resolved_source)

        # Also try matching original source string if path resolution differed
        if removed_count == 0 and resolved_source != source:
            removed_count = self.vector_store.delete_document(source)
            bm25_removed = self.bm25_index.remove_document(source)

        success = removed_count > 0 or bm25_removed > 0
        message = (
            f"Removed {removed_count} chunk(s) from vector store and {bm25_removed} from BM25 index for '{source}'."
            if success else f"No indexed chunks found for '{source}'."
        )

        return {
            "success": success,
            "source": source,
            "chunks_removed": removed_count,
            "bm25_chunks_removed": bm25_removed,
            "message": message,
        }

    def reindex_document(self, source: str) -> Dict[str, Any]:
        """
        Re-index an existing document or URL (removes old chunks and indexes fresh content).

        Args:
            source: File path or URL.

        Returns:
            Dict with re-indexing status.
        """
        logger.info("Re-indexing document source: %s", source)
        self.remove_document(source)
        return self.index_document(source)

    def index_directory(self, dir_path: str, recursive: bool = True) -> List[Dict[str, Any]]:
        """
        Batch index all supported documents inside a directory.

        Args:
            dir_path: Directory path containing documents.
            recursive: Whether to scan subdirectories recursively.

        Returns:
            List of indexing result dicts.
        """
        path = Path(dir_path).resolve()
        if not path.exists() or not path.is_dir():
            raise ValueError(f"Directory not found: {dir_path}")

        supported_exts = RAGDocumentLoader.SUPPORTED_EXTENSIONS.keys()
        pattern = "**/*" if recursive else "*"
        
        results = []
        for file_item in path.glob(pattern):
            if file_item.is_file() and file_item.suffix.lower() in supported_exts:
                res = self.index_document(str(file_item))
                results.append(res)

        logger.info("Directory indexing complete for '%s' (%d file(s) processed).", dir_path, len(results))
        return results

    def list_documents(self) -> List[Dict[str, Any]]:
        """List all currently indexed knowledge sources."""
        return self.vector_store.list_documents()

    def clear_all(self) -> int:
        """Clear the entire knowledge base (both vector store and BM25 index)."""
        vector_count = self.vector_store.clear_all()
        bm25_count = self.bm25_index.clear_all()
        logger.info("Cleared entire knowledge base: %d chunks from vector store, %d from BM25 index.", vector_count, bm25_count)
        return vector_count
