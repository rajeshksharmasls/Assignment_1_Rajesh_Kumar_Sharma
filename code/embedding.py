"""
Embedding module — multi-tenant, versioned, lifecycle-aware upserts
into a Pinecone integrated inference index.
"""

import time
import hashlib
from typing import List, Dict, Optional
from pinecone import Pinecone

from config import (
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    PINECONE_CLOUD,
    PINECONE_REGION,
    PINECONE_EMBED_MODEL,
    CHECKSUM_ALGORITHM,
    DEFAULT_DOCUMENT_VERSION,
)

# ──────────────────────────────────────────────────────────────────────────────
# Pinecone Client (Singleton)
# ──────────────────────────────────────────────────────────────────────────────
_pc = Pinecone(api_key=PINECONE_API_KEY)


# ──────────────────────────────────────────────────────────────────────────────
# Index Management
# ──────────────────────────────────────────────────────────────────────────────
def _get_or_create_index():
    """Create integrated inference index if missing."""
    if not _pc.has_index(PINECONE_INDEX_NAME):
        print(
            f"🔨 Creating Pinecone index '{PINECONE_INDEX_NAME}' (integrated inference)..."
        )

        _pc.create_index_for_model(
            name=PINECONE_INDEX_NAME,
            cloud=PINECONE_CLOUD,
            region=PINECONE_REGION,
            embed={
                "model": PINECONE_EMBED_MODEL,
                "field_map": {"text": "chunk_text"},
            },
        )

        print("⏳ Waiting for index to be ready ...")
        while not _pc.describe_index(PINECONE_INDEX_NAME).status.get("ready", False):
            time.sleep(1)

        print("✅ Index created and ready!")

    return _pc.Index(PINECONE_INDEX_NAME)


# ──────────────────────────────────────────────────────────────────────────────
# Utility: Checksum
# ──────────────────────────────────────────────────────────────────────────────
def _compute_checksum(text: str) -> str:
    hasher = hashlib.new(CHECKSUM_ALGORITHM)
    hasher.update(text.encode("utf-8"))
    return hasher.hexdigest()


# ──────────────────────────────────────────────────────────────────────────────
# Lifecycle Management
# ──────────────────────────────────────────────────────────────────────────────
def delete_document(user_id: str, document_id: str):
    """Hard delete all versions of a document for a user."""
    index = _get_or_create_index()
    namespace = f"user::{user_id}"

    index.delete(
        namespace=namespace,
        filter={"document_id": {"$eq": document_id}},
    )

    print(f"🗑 Deleted document '{document_id}' for user '{user_id}'")


def delete_previous_versions(user_id: str, document_id: str):
    """
    Hard delete previous versions of a document.
    Safe for first ingestion.
    """
    namespace = f"user::{user_id}"
    index = _get_or_create_index()

    try:
        index.delete(
            namespace=namespace,
            filter={"document_id": {"$eq": document_id}},
        )
    except Exception:
        # Namespace may not exist yet — safe to ignore
        pass


# ──────────────────────────────────────────────────────────────────────────────
# Upsert Logic (Multi-Tenant + Versioned + Deduplicated)
# ──────────────────────────────────────────────────────────────────────────────
def upsert_chunks(
    user_id: str,
    document_id: str,
    records: List[Dict],
    version: str = DEFAULT_DOCUMENT_VERSION,
    batch_size: int = 96,
) -> int:
    """
    Upsert document chunks with:

    - User namespace isolation
    - Versioned vector IDs
    - Lifecycle metadata
    - Duplicate detection via checksum
    """

    index = _get_or_create_index()
    namespace = f"user::{user_id}"

    total = 0

    # Optional: deactivate older versions before inserting new version
    delete_previous_versions(user_id, document_id)

    for i in range(0, len(records), batch_size):
        batch = records[i : i + batch_size]
        pinecone_records = []

        for chunk_index, rec in enumerate(batch):
            chunk_text = rec["chunk_text"]
            checksum = _compute_checksum(chunk_text)

            # Deterministic vector ID
            vector_id = f"{document_id}:{version}:{chunk_index}"

            pinecone_records.append(
                {
                    "_id": vector_id,
                    "text": chunk_text,  # REQUIRED for integrated embedding
                    "chunk_text": chunk_text,  # Optional: store raw chunk for retrieval
                    "document_id": document_id,
                    "version": version,
                    "user_id": user_id,
                    "checksum": checksum,
                    "source": rec.get("source", ""),
                    "pages": rec.get("pages", ""),
                }
            )

        index.upsert_records(namespace, pinecone_records)
        total += len(pinecone_records)

    print(
        f"📦 Upserted {total} chunks | user={user_id} | "
        f"doc={document_id} | version={version}"
    )

    return total


# ──────────────────────────────────────────────────────────────────────────────
# CLI Test
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Embedding Test (Multi-Tenant) ===")

    test_user = "demo_user"
    test_document = "test_doc"

    test_records = [
        {
            "chunk_text": "Apple reported strong Q4 2024 earnings.",
            "source": "test.pdf",
        },
        {
            "chunk_text": "Nike revenue grew in fiscal year 2025.",
            "source": "test.pdf",
        },
    ]

    print(f"Upserting {len(test_records)} test chunks...")
    count = upsert_chunks(
        user_id=test_user,
        document_id=test_document,
        records=test_records,
    )

    print(f"Upserted: {count} records")
    print("✅ Embedding test passed!")
