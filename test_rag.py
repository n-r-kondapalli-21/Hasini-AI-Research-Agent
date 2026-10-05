"""
Unit tests for Hasini RAG Knowledge System.

Tests document loading, text chunking, ChromaDB vector store operations,
BM25 keyword search, RRF fusion, cross-encoder reranking,
relevance threshold filtering, deletion, empty results, and indexer lifecycle.
"""

import os
import tempfile
import unittest
from pathlib import Path

from rag.document_loader import RAGDocumentLoader
from rag.text_splitter import RAGTextSplitter
from rag.vector_store import RAGVectorStore
from rag.indexer import RAGIndexer
from rag.retriever import RAGRetriever
from rag.bm25_index import BM25Index
from rag.fusion import reciprocal_rank_fusion, filter_by_relevance


class TestRAGSystem(unittest.TestCase):

    def setUp(self):
        """Set up temporary directory for ChromaDB and sample files."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test_chroma_db")
        self.bm25_path = os.path.join(self.temp_dir.name, "test_bm25_index")

        # Initialize vector store with temporary path
        self.vector_store = RAGVectorStore(db_path=self.db_path)
        self.bm25_index = BM25Index(index_path=self.bm25_path)
        self.indexer = RAGIndexer(
            vector_store=self.vector_store,
            bm25_index=self.bm25_index,
            chunk_size=200,
            chunk_overlap=20,
        )
        self.retriever = RAGRetriever(
            vector_store=self.vector_store,
            bm25_index=self.bm25_index,
        )

        # Create sample test text file
        self.sample_txt = os.path.join(self.temp_dir.name, "sample_research.txt")
        with open(self.sample_txt, "w", encoding="utf-8") as f:
            f.write(
                "Hasini is an advanced AI research agent built for multi-modal operations. "
                "It supports local tool execution, web search, MCP servers, and RAG knowledge systems. "
                "Quantum computing uses qubits to perform complex calculations exponentially faster than classical computers."
            )

        # Create sample markdown file
        self.sample_md = os.path.join(self.temp_dir.name, "notes.md")
        with open(self.sample_md, "w", encoding="utf-8") as f:
            f.write(
                "# Machine Learning Notes\n"
                "Supervised learning algorithms use labeled datasets to train models. "
                "Unsupervised learning finds hidden patterns in unlabeled data."
            )

    def tearDown(self):
        """Clean up temporary directory."""
        try:
            self.temp_dir.cleanup()
        except Exception:
            pass

    def test_document_loader(self):
        """Test loading plain text and markdown documents."""
        txt_doc = RAGDocumentLoader.load(self.sample_txt)
        self.assertEqual(txt_doc["file_type"], "txt")
        self.assertIn("Hasini is an advanced AI research agent", txt_doc["content"])
        self.assertEqual(txt_doc["filename"], "sample_research.txt")

        md_doc = RAGDocumentLoader.load(self.sample_md)
        self.assertEqual(md_doc["file_type"], "markdown")
        self.assertIn("Supervised learning", md_doc["content"])

    def test_text_splitter(self):
        """Test text chunking with metadata."""
        splitter = RAGTextSplitter(chunk_size=100, chunk_overlap=10)
        doc = RAGDocumentLoader.load(self.sample_txt)
        chunks = splitter.create_chunks_with_metadata(doc, timestamp="2026-10-04T12:00:00Z")

        self.assertGreater(len(chunks), 0)
        first_chunk = chunks[0]
        self.assertIn("id", first_chunk)
        self.assertIn("text", first_chunk)
        self.assertEqual(first_chunk["metadata"]["filename"], "sample_research.txt")
        self.assertEqual(first_chunk["metadata"]["timestamp"], "2026-10-04T12:00:00Z")

    def test_indexing_and_retrieval(self):
        """Test document indexing into ChromaDB and retrieving relevant context."""
        res = self.indexer.index_document(self.sample_txt)
        self.assertTrue(res["success"])
        self.assertGreater(res["chunks_indexed"], 0)

        # Search for quantum computing query
        query_results = self.retriever.retrieve(
            query="Tell me about quantum computing and qubits",
            top_k=3,
            threshold=0.1,
            enabled=True,
        )

        self.assertGreater(len(query_results), 0)
        top_result = query_results[0]
        self.assertIn("qubits", top_result["text"].lower())
        self.assertEqual(top_result["metadata"]["filename"], "sample_research.txt")

    def test_similarity_threshold_filtering(self):
        """Test that chunks below similarity threshold are filtered out."""
        self.indexer.index_document(self.sample_txt)

        # High similarity threshold for an unrelated topic (e.g., baking recipes)
        strict_results = self.retriever.retrieve(
            query="recipe for baking chocolate banana cake with sugar",
            top_k=3,
            threshold=0.9, # Very high threshold
            enabled=True,
        )

        # Should be empty or filtered out because content is about AI and quantum computing
        self.assertEqual(len(strict_results), 0)

    def test_deletion(self):
        """Test deleting a document and confirming chunk removal."""
        self.indexer.index_document(self.sample_txt)
        docs_before = self.indexer.list_documents()
        self.assertEqual(len(docs_before), 1)

        rem_res = self.indexer.remove_document(self.sample_txt)
        self.assertTrue(rem_res["success"])

        docs_after = self.indexer.list_documents()
        self.assertEqual(len(docs_after), 0)

    def test_empty_results_and_disabled_rag(self):
        """Test querying empty store or when RAG is disabled."""
        # Empty vector store
        empty_res = self.retriever.retrieve(query="anything", enabled=True)
        self.assertEqual(len(empty_res), 0)

        # Disabled RAG
        self.indexer.index_document(self.sample_txt)
        disabled_res = self.retriever.retrieve(query="Hasini AI", enabled=False)
        self.assertEqual(len(disabled_res), 0)

    def test_reindex_and_list(self):
        """Test listing documents and reindexing."""
        self.indexer.index_document(self.sample_txt)
        self.indexer.index_document(self.sample_md)

        docs = self.indexer.list_documents()
        self.assertEqual(len(docs), 2)

        # Reindex sample_txt
        reindex_res = self.indexer.reindex_document(self.sample_txt)
        self.assertTrue(reindex_res["success"])

        docs_after = self.indexer.list_documents()
        self.assertEqual(len(docs_after), 2)

    def test_bm25_indexing_and_search(self):
        """Test BM25 keyword indexing and search."""
        # Index document
        res = self.indexer.index_document(self.sample_txt)
        self.assertTrue(res["success"])

        # BM25 search for keyword
        bm25_results = self.bm25_index.search(query="quantum computing", top_k=3)
        self.assertGreater(len(bm25_results), 0)
        self.assertIn("score", bm25_results[0])
        self.assertIn("rank", bm25_results[0])

    def test_rrf_fusion(self):
        """Test Reciprocal Rank Fusion of vector and BM25 results."""
        # Index document
        self.indexer.index_document(self.sample_txt)

        # Get vector results
        vector_results = self.vector_store.query(query="quantum computing", top_k=5)
        self.assertGreater(len(vector_results), 0)

        # Get BM25 results
        bm25_results = self.bm25_index.search(query="quantum computing", top_k=5)
        self.assertGreater(len(bm25_results), 0)

        # Fuse results
        fused = reciprocal_rank_fusion(vector_results, bm25_results, k=60)
        self.assertGreater(len(fused), 0)
        self.assertIn("rrf_score", fused[0])

        # Check that fused results are sorted by RRF score
        for i in range(len(fused) - 1):
            self.assertGreaterEqual(fused[i]["rrf_score"], fused[i + 1]["rrf_score"])

    def test_rrf_with_empty_results(self):
        """Test RRF fusion when one or both result lists are empty."""
        # Only vector results
        vector_results = [{"id": "1", "text": "test", "metadata": {}, "similarity": 0.8}]
        bm25_results = []
        fused = reciprocal_rank_fusion(vector_results, bm25_results)
        self.assertEqual(len(fused), 1)

        # Only BM25 results
        vector_results = []
        bm25_results = [{"chunk_id": "1", "score": 1.5, "rank": 1}]
        fused = reciprocal_rank_fusion(vector_results, bm25_results)
        self.assertEqual(len(fused), 1)

        # Both empty
        fused = reciprocal_rank_fusion([], [])
        self.assertEqual(len(fused), 0)

    def test_relevance_filtering(self):
        """Test relevance score filtering."""
        chunks = [
            {"id": "1", "text": "test1", "metadata": {}, "similarity": 0.9},
            {"id": "2", "text": "test2", "metadata": {}, "similarity": 0.7},
            {"id": "3", "text": "test3", "metadata": {}, "similarity": 0.4},
            {"id": "4", "text": "test4", "metadata": {}, "similarity": 0.2},
        ]

        # Filter with threshold 0.5
        filtered = filter_by_relevance(chunks, threshold=0.5, score_key="similarity")
        self.assertEqual(len(filtered), 2)
        self.assertEqual(filtered[0]["id"], "1")
        self.assertEqual(filtered[1]["id"], "2")

        # No filtering with threshold 0
        filtered = filter_by_relevance(chunks, threshold=0.0, score_key="similarity")
        self.assertEqual(len(filtered), 4)

    def test_hybrid_retrieval(self):
        """Test complete hybrid retrieval pipeline."""
        # Index documents
        self.indexer.index_document(self.sample_txt)
        self.indexer.index_document(self.sample_md)

        # Hybrid search
        results = self.retriever.retrieve(
            query="quantum computing and machine learning",
            top_k=3,
            threshold=0.1,
            enabled=True,
        )

        self.assertGreater(len(results), 0)
        # Check that results have expected fields
        for result in results:
            self.assertIn("id", result)
            self.assertIn("text", result)
            self.assertIn("metadata", result)
            self.assertIn("similarity", result)

    def test_bm25_disabled(self):
        """Test retrieval when BM25 is disabled."""
        # Index document
        self.indexer.index_document(self.sample_txt)

        # Temporarily disable BM25 by clearing index
        self.bm25_index.clear_all()

        # Should still work with vector-only search
        results = self.retriever.retrieve(
            query="quantum computing",
            top_k=3,
            threshold=0.1,
            enabled=True,
        )

        self.assertGreater(len(results), 0)

    def test_irrelevant_query_filtering(self):
        """Test that irrelevant queries return no results with high threshold."""
        self.indexer.index_document(self.sample_txt)

        # Query for completely unrelated topic with high threshold
        results = self.retriever.retrieve(
            query="recipe for chocolate cake baking",
            top_k=3,
            threshold=0.9,
            enabled=True,
        )

        # Should be empty or very few results
        self.assertEqual(len(results), 0)


if __name__ == "__main__":
    unittest.main()
