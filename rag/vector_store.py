"""
Vector store module for RAG Knowledge System.

Manages persistent ChromaDB vector collection storage, indexing, querying, and deletion.
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import chromadb
from chromadb.config import Settings

from config import CHROMADB_PATH
from .embedding import get_embedding_function

logger = logging.getLogger("hasini.rag.vector_store")


class RAGVectorStore:
    """
    Wrapper around persistent ChromaDB for RAG knowledge storage and retrieval.
    """

    COLLECTION_NAME = "hasini_knowledge_base"

    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize persistent ChromaDB client and collection.

        Args:
            db_path: Path to persistent directory. Defaults to config CHROMADB_PATH.
        """
        self.db_path = str(Path(db_path or CHROMADB_PATH).resolve())
        os.makedirs(self.db_path, exist_ok=True)

        logger.info("Initializing ChromaDB persistent store at: %s", self.db_path)

        self.client = chromadb.PersistentClient(
            path=self.db_path,
            settings=Settings(anonymized_telemetry=False),
        )

        self.embedding_fn = get_embedding_function()

        # Initialize collection with cosine similarity space
        try:
            self.collection = self.client.get_or_create_collection(
                name=self.COLLECTION_NAME,
                embedding_function=self.embedding_fn,
                metadata={"hnsw:space": "cosine"},
            )
        except ValueError as val_err:
            if "embedding function" in str(val_err).lower() or "conflict" in str(val_err).lower():
                logger.warning(
                    "Embedding function conflict detected for collection '%s'. "
                    "Recreating collection with local embedding function...",
                    self.COLLECTION_NAME,
                )
                try:
                    self.client.delete_collection(name=self.COLLECTION_NAME)
                except Exception:
                    pass
                self.collection = self.client.create_collection(
                    name=self.COLLECTION_NAME,
                    embedding_function=self.embedding_fn,
                    metadata={"hnsw:space": "cosine"},
                )
            else:
                raise

        logger.info(
            "ChromaDB collection '%s' ready (%d chunk(s) stored).",
            self.COLLECTION_NAME,
            self.collection.count(),
        )

    def add_chunks(self, chunks: List[Dict[str, Any]]) -> int:
        """
        Add or update chunk documents in the ChromaDB collection.

        Args:
            chunks: List of chunk dicts containing 'id', 'text', and 'metadata'.

        Returns:
            Number of chunks successfully added.
        """
        if not chunks:
            return 0

        ids = [c["id"] for c in chunks]
        documents = [c["text"] for c in chunks]
        metadatas = [c["metadata"] for c in chunks]

        try:
            self.collection.upsert(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
            )
            logger.info("Upserted %d chunk(s) into ChromaDB.", len(chunks))
            return len(chunks)
        except Exception as exc:
            logger.error("Failed to add chunks to ChromaDB: %s", exc)
            raise

    def query(self, query_text: str = "", top_k: int = 5, query: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Query the vector store for top-k similar chunks.

        Args:
            query_text: Search query string (or pass via `query`).
            top_k: Number of nearest chunks to retrieve.

        Returns:
            List of dicts containing 'id', 'text', 'metadata', 'distance', and 'similarity'.
        """
        search_text = (query_text or query or "").strip()
        if not search_text:
            return []

        total_items = self.collection.count()
        if total_items == 0:
            logger.debug("Vector store is empty. Returning 0 query results.")
            return []

        actual_k = min(top_k, total_items)

        try:
            results = self.collection.query(
                query_texts=[search_text],
                n_results=actual_k,
                include=["documents", "metadatas", "distances"],
            )

            formatted_results = []
            if results and results.get("documents") and results["documents"][0]:
                docs = results["documents"][0]
                metas = results["metadatas"][0] if results.get("metadatas") else [{}] * len(docs)
                dists = results["distances"][0] if results.get("distances") else [0.0] * len(docs)
                ids = results["ids"][0] if results.get("ids") else [""] * len(docs)

                for doc_text, meta, dist, chunk_id in zip(docs, metas, dists, ids):
                    # For Cosine distance space: distance = 1.0 - cosine_similarity
                    # Similarity range: 1.0 (identical) to 0.0 (orthogonal) or negative (opposite)
                    similarity = max(0.0, 1.0 - float(dist)) if dist is not None else 0.0

                    formatted_results.append({
                        "id": chunk_id,
                        "text": doc_text,
                        "metadata": meta,
                        "distance": float(dist) if dist is not None else 0.0,
                        "similarity": round(similarity, 4),
                    })

            return formatted_results

        except Exception as exc:
            logger.error("Error querying ChromaDB vector store: %s", exc)
            return []

    def delete_document(self, source: str) -> int:
        """
        Delete all chunks belonging to a specific document source path or URL.

        Args:
            source: Source path or URL matching document metadata.

        Returns:
            Number of chunks removed (approximate based on count diff).
        """
        initial_count = self.collection.count()
        if initial_count == 0:
            return 0

        try:
            self.collection.delete(where={"source": source})
            new_count = self.collection.count()
            removed = initial_count - new_count
            logger.info("Deleted document '%s' (%d chunk(s) removed).", source, removed)
            return removed
        except Exception as exc:
            logger.error("Failed to delete document '%s': %s", source, exc)
            return 0

    def list_documents(self) -> List[Dict[str, Any]]:
        """
        Get a list of all unique documents currently indexed in the knowledge store.

        Returns:
            List of dicts with 'source', 'filename', 'file_type', 'chunk_count', 'timestamp'.
        """
        if self.collection.count() == 0:
            return []

        try:
            data = self.collection.get(include=["metadatas"])
            metadatas = data.get("metadatas", [])

            docs_summary: Dict[str, Dict[str, Any]] = {}
            for meta in metadatas:
                if not meta or "source" not in meta:
                    continue

                source = meta["source"]
                if source not in docs_summary:
                    docs_summary[source] = {
                        "source": source,
                        "filename": meta.get("filename", os.path.basename(source)),
                        "file_type": meta.get("file_type", "unknown"),
                        "chunk_count": 0,
                        "timestamp": meta.get("timestamp", "N/A"),
                    }
                docs_summary[source]["chunk_count"] += 1

            return list(docs_summary.values())

        except Exception as exc:
            logger.error("Failed to list indexed documents: %s", exc)
            return []

    def clear_all(self) -> int:
        """Delete all items from the ChromaDB knowledge base collection."""
        count = self.collection.count()
        if count > 0:
            self.client.delete_collection(name=self.COLLECTION_NAME)
            self.collection = self.client.get_or_create_collection(
                name=self.COLLECTION_NAME,
                embedding_function=self.embedding_fn,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info("Cleared all %d item(s) from ChromaDB collection.", count)
        return count

    def get_stats(self) -> Dict[str, Any]:
        """Return system statistics about the vector store."""
        total_chunks = self.collection.count()
        docs = self.list_documents()
        return {
            "total_chunks": total_chunks,
            "total_documents": len(docs),
            "collection_name": self.COLLECTION_NAME,
            "db_path": self.db_path,
        }
