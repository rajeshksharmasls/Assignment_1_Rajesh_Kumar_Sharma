"""
Ingestion module — multi-user, version-aware document ingestion.
Supports:
- User-level isolation
- Document lifecycle metadata
- Version tracking
- Duplicate detection support
"""

import os
import re
import hashlib
from typing import List, Dict
from pypdf import PdfReader

from config import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    CHECKSUM_ALGORITHM,
    DEFAULT_DOCUMENT_VERSION,
)


# ─────────────────────────────────────────────────────────────
# Utility: Compute checksum (Duplicate Detection)
# ─────────────────────────────────────────────────────────────


def compute_checksum(file_path: str) -> str:
    """Generate checksum for duplicate detection."""
    hasher = hashlib.new(CHECKSUM_ALGORITHM)
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()


# ─────────────────────────────────────────────────────────────
# Text Extraction
# ─────────────────────────────────────────────────────────────


def extract_pages_from_pdf(file_path: str) -> List[Dict]:
    reader = PdfReader(file_path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            pages.append({"page": i + 1, "text": text})
    return pages


def extract_pages_from_txt(file_path: str) -> List[Dict]:
    with open(file_path, "r", encoding="utf-8") as f:
        return [{"page": 1, "text": f.read()}]


def extract_pages(file_path: str) -> List[Dict]:
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return extract_pages_from_pdf(file_path)
    elif ext in (".txt", ".md"):
        return extract_pages_from_txt(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")


# ─────────────────────────────────────────────────────────────
# Chunking
# ─────────────────────────────────────────────────────────────


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def chunk_pages(
    pages: List[Dict],
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> List[Dict]:

    full_text = ""
    char_to_page: List[int] = []

    for p in pages:
        cleaned = clean_text(p["text"])
        if cleaned:
            if full_text:
                full_text += " "
                char_to_page.append(p["page"])
            full_text += cleaned
            char_to_page.extend([p["page"]] * len(cleaned))

    chunks: List[Dict] = []
    start = 0

    while start < len(full_text):
        end = min(start + chunk_size, len(full_text))
        chunk = full_text[start:end].strip()

        if chunk:
            page_set = sorted(set(char_to_page[start:end]))
            chunks.append({"chunk_text": chunk, "pages": page_set})

        start += chunk_size - overlap

    return chunks


# ─────────────────────────────────────────────────────────────
# Multi-User / Version-Aware Ingestion
# ─────────────────────────────────────────────────────────────


def ingest_document(
    file_path: str,
    user_id: str,
    doc_id: str,
    version: int = DEFAULT_DOCUMENT_VERSION,
) -> List[Dict]:
    """
    Full ingestion pipeline for a single document.
    Multi-user + version aware.

    Returns:
    [
        {
            id,
            chunk_text,
            source,
            pages,
            metadata
        }
    ]
    """

    file_name = os.path.basename(file_path)
    checksum = compute_checksum(file_path)
    pages = extract_pages(file_path)
    chunks = chunk_pages(pages)

    records = []

    for idx, chunk in enumerate(chunks):
        page_str = ",".join(str(p) for p in chunk["pages"])

        # Deterministic chunk ID (important for version replacement)
        chunk_id = f"{doc_id}::v{version}::chunk-{idx}"

        records.append(
            {
                "id": chunk_id,
                "chunk_text": chunk["chunk_text"],
                "source": file_name,
                "pages": page_str,
                "metadata": {
                    "user_id": user_id,
                    "doc_id": doc_id,
                    "version": version,
                    "checksum": checksum,
                    "status": "active",  # lifecycle state
                },
            }
        )

    print(
        f"✅ Ingested '{file_name}' "
        f"(User: {user_id}, Version: {version}) "
        f"→ {len(records)} chunks"
    )

    return records


# ─────────────────────────────────────────────────────────────
# CLI Test
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import uuid

    print("=== Multi-User Ingestion Test ===")

    test_path = (
        sys.argv[1] if len(sys.argv) > 1 else "docs/Student-Handbook-2025-2026.pdf"
    )
    test_user = "user_123"
    test_doc_id = str(uuid.uuid4())

    records = ingest_document(
        file_path=test_path,
        user_id=test_user,
        doc_id=test_doc_id,
        version=1,
    )

    print(f"Total chunks: {len(records)}")
    print("\nFirst chunk preview:")
    print(f"  ID      : {records[0]['id']}")
    print(f"  Metadata: {records[0]['metadata']}")
    print(f"  Text    : {records[0]['chunk_text'][:200]}...")
    print("✅ Ingestion test passed!")
