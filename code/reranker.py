"""
Reranker module — multi-tenant, lifecycle-aware semantic search
with Pinecone integrated embedding + hosted reranker.
"""

from typing import List, Dict, Optional
from pinecone import Pinecone

from config import (
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    PINECONE_RERANK_MODEL,
    RERANK_TOP_N,
    TOP_K,
)

_pc = Pinecone(api_key=PINECONE_API_KEY)


# ──────────────────────────────────────────────────────────────────────────────
# Safe Index Access
# ──────────────────────────────────────────────────────────────────────────────
def _get_index():
    if not _pc.has_index(PINECONE_INDEX_NAME):
        raise ValueError(f"Pinecone index '{PINECONE_INDEX_NAME}' does not exist.")
    return _pc.Index(PINECONE_INDEX_NAME)


# ──────────────────────────────────────────────────────────────────────────────
# Reranked Search
# ──────────────────────────────────────────────────────────────────────────────
def rerank(
    user_id: str,
    query: str,
    top_k: int = TOP_K,
    top_n: int = RERANK_TOP_N,
    document_id: Optional[str] = None,
    version: Optional[str] = None,
) -> List[Dict]:
    """
    Multi-tenant reranked semantic search.

    Features:
    - User-level namespace isolation
    - Lifecycle filtering (active only)
    - Optional document filter
    - Optional version filter
    - Integrated embedding + hosted reranker
    """

    index = _get_index()
    namespace = f"user::{user_id}"

    # ── Metadata Filters ──────────────────────────────────────────────────────
    metadata_filter = {
        "lifecycle_status": {"$eq": "active"},
    }

    if document_id:
        metadata_filter["document_id"] = {"$eq": document_id}

    if version:
        metadata_filter["version"] = {"$eq": version}

    reranked = index.search(
        namespace=namespace,
        query={
            "top_k": top_k,
            "inputs": {"text": query},
            "filter": metadata_filter,
        },
        rerank={
            "model": PINECONE_RERANK_MODEL,
            "top_n": top_n,
            "rank_fields": ["chunk_text"],
        },
        fields=[
            "chunk_text",
            "source",
            "pages",
            "document_id",
            "version",
            "checksum",
        ],
    )

    hits = []
    for item in reranked.get("result", {}).get("hits", []):
        hits.append(
            {
                "id": item.get("_id", ""),
                "score": item.get("_score", 0.0),
                "chunk_text": item.get("fields", {}).get("chunk_text", ""),
                "source": item.get("fields", {}).get("source", ""),
                "pages": item.get("fields", {}).get("pages", ""),
                "document_id": item.get("fields", {}).get("document_id", ""),
                "version": item.get("fields", {}).get("version", ""),
                "checksum": item.get("fields", {}).get("checksum", ""),
            }
        )

    return hits


# ──────────────────────────────────────────────────────────────────────────────
# CLI Test
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    print("=== Reranker Test (Multi-Tenant) ===")

    user_id = "demo_user"
    query = (
        " ".join(sys.argv[1:])
        if len(sys.argv) > 1
        else "What is the Teaching Schedule for Year 1?"
    )

    print(f"User: {user_id}")
    print(f"Query: {query}\n")

    results = rerank(
        user_id=user_id,
        query=query,
        top_k=5,
        top_n=3,
    )

    print(f"Reranked {len(results)} results:\n")

    for i, r in enumerate(results, 1):
        print(
            f"[{i}] score={r['score']:.4f} | "
            f"doc={r['document_id']} | "
            f"v={r['version']} | "
            f"source={r['source']}"
        )
        print(f"    {r['chunk_text'][:150]}...\n")

    print("✅ Reranker test passed!")
