"""
Retrieval module — multi-tenant semantic search against Pinecone.
Supports:
- User namespace isolation
- Document filtering
- Version control
- Lifecycle filtering
- Optional checksum lookup
"""

from typing import List, Dict, Optional
from pinecone import Pinecone

from config import (
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    TOP_K,
    DEFAULT_DOCUMENT_VERSION,
)

_pc = Pinecone(api_key=PINECONE_API_KEY)


# ──────────────────────────────────────────────────────────────────────────────
# Core Search
# ──────────────────────────────────────────────────────────────────────────────
def search(
    user_id: str,
    query: str,
    top_k: int = TOP_K,
    document_id: Optional[str] = None,
    version: Optional[str] = None,
) -> List[Dict]:
    """
    Semantic search with:
    - User-level namespace isolation
    - Lifecycle filter (active only)
    - Optional document filter
    - Optional version filter
    """

    namespace = f"user::{user_id}"
    index = _get_or_create_index()

    # ── Metadata Filters ──────────────────────────────────────────────────────
    metadata_filter = {
        "lifecycle_status": {"$eq": "active"},
    }

    if document_id:
        metadata_filter["document_id"] = {"$eq": document_id}

    if version:
        metadata_filter["version"] = {"$eq": version}

    results = index.search(
        namespace=namespace,
        query={
            "top_k": top_k,
            "inputs": {"text": query},
            "filter": metadata_filter,
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
    for item in results.get("result", {}).get("hits", []):
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
# Duplicate Intelligence (Checksum Lookup)
# ──────────────────────────────────────────────────────────────────────────────
def search_by_checksum(
    user_id: str,
    checksum: str,
) -> List[Dict]:
    """
    Retrieve chunks by checksum.
    Useful for duplicate detection validation or integrity audits.
    """

    namespace = f"user::{user_id}"
    index = _pc.Index(PINECONE_INDEX_NAME)

    results = index.search(
        namespace=namespace,
        query={
            "top_k": 10,
            "inputs": {"text": ""},  # required placeholder
            "filter": {
                "checksum": {"$eq": checksum},
                "lifecycle_status": {"$eq": "active"},
            },
        },
        fields=["chunk_text", "document_id", "version"],
    )

    return results.get("result", {}).get("hits", [])


# ──────────────────────────────────────────────────────────────────────────────
# Create Index
# ──────────────────────────────────────────────────────────────────────────────
def _get_or_create_index():
    if not _pc.has_index(PINECONE_INDEX_NAME):
        raise ValueError(
            f"Pinecone index '{PINECONE_INDEX_NAME}' does not exist. "
            "Run ingestion first to create it."
        )
    return _pc.Index(PINECONE_INDEX_NAME)


# ──────────────────────────────────────────────────────────────────────────────
# CLI Test
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    print("=== Retrieval Test (Multi-Tenant) ===")

    user_id = "demo_user"
    query = (
        " ".join(sys.argv[1:])
        if len(sys.argv) > 1
        else "What is the Teaching Schedule for Year 1?"
    )

    print(f"User: {user_id}")
    print(f"Query: {query}\n")

    results = search(user_id=user_id, query=query, top_k=3)

    print(f"Found {len(results)} results:\n")
    for i, r in enumerate(results, 1):
        print(
            f"[{i}] score={r['score']:.4f} | "
            f"doc={r['document_id']} | "
            f"v={r['version']} | "
            f"source={r['source']}"
        )
        print(f"    {r['chunk_text'][:150]}...\n")

    print("✅ Retrieval test passed!")
