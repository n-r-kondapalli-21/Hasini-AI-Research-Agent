"""
Result fusion module for RAG Knowledge System.

Implements Reciprocal Rank Fusion (RRF) to combine vector and BM25 search results.
"""

import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger("hasini.rag.fusion")


def reciprocal_rank_fusion(
    vector_results: List[Dict[str, Any]],
    bm25_results: List[Dict[str, Any]],
    k: int = 60,
) -> List[Dict[str, Any]]:
    """
    Combine vector and BM25 results using Reciprocal Rank Fusion (RRF).

    RRF formula: score = sum(1 / (k + rank)) for each result list

    Args:
        vector_results: List of vector search results with 'id' and 'similarity'.
        bm25_results: List of BM25 results with 'chunk_id' and 'score'.
        k: RRF constant (default 60, as per original paper).

    Returns:
        List of fused results with combined RRF scores, sorted by score descending.
    """
    if not vector_results and not bm25_results:
        logger.debug("Both vector and BM25 results are empty. Returning empty list.")
        return []

    # Initialize score map
    rrf_scores: Dict[str, float] = {}
    result_map: Dict[str, Dict[str, Any]] = {}

    # Process vector results
    for rank, result in enumerate(vector_results, 1):
        chunk_id = result["id"]
        score = 1.0 / (k + rank)
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + score
        result_map[chunk_id] = result

    # Process BM25 results
    for rank, result in enumerate(bm25_results, 1):
        chunk_id = result["chunk_id"]
        score = 1.0 / (k + rank)
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + score
        # Store BM25 result if not already in map
        if chunk_id not in result_map:
            result_map[chunk_id] = {
                "id": chunk_id,
                "text": "",  # Will be filled later from vector store
                "metadata": {},
                "similarity": 0.0,
                "distance": 0.0,
            }

    # Combine and sort by RRF score
    fused_results = []
    for chunk_id, rrf_score in rrf_scores.items():
        result = result_map[chunk_id].copy()
        result["rrf_score"] = round(rrf_score, 4)
        fused_results.append(result)

    fused_results.sort(key=lambda x: x["rrf_score"], reverse=True)

    logger.info(
        "RRF fusion: %d vector results, %d BM25 results, %d fused results.",
        len(vector_results),
        len(bm25_results),
        len(fused_results),
    )

    return fused_results


def filter_by_relevance(
    chunks: List[Dict[str, Any]],
    threshold: float = 0.0,
    score_key: str = "similarity",
) -> List[Dict[str, Any]]:
    """
    Filter chunks by relevance score threshold.

    Args:
        chunks: List of chunk dicts with score field.
        threshold: Minimum score threshold.
        score_key: Key to use for scoring (e.g., 'similarity', 'rerank_score', 'rrf_score').

    Returns:
        Filtered list of chunks.
    """
    if threshold <= 0.0:
        return chunks

    filtered = [c for c in chunks if c.get(score_key, 0.0) >= threshold]

    logger.debug(
        "Relevance filtering: %d -> %d chunks (threshold=%.2f, key=%s).",
        len(chunks),
        len(filtered),
        threshold,
        score_key,
    )

    return filtered
