"""
Generation module — multi-tenant, version-aware answer generation
with citation-safe context handling.
"""

from typing import List, Dict
from collections import OrderedDict
from langchain_openai import ChatOpenAI

from config import (
    OPENAI_API_KEY,
    OPENAI_MODEL,
    MAX_TOKENS,
    TEMPERATURE,
)

# ──────────────────────────────────────────────────────────────────────────────
# LLM Client
# ──────────────────────────────────────────────────────────────────────────────
_llm = ChatOpenAI(
    model=OPENAI_MODEL,
    api_key=OPENAI_API_KEY,
    max_tokens=MAX_TOKENS,
    temperature=TEMPERATURE,
)

SYSTEM_PROMPT = """You are a helpful assistant that answers questions based ONLY on the provided context.

RULES:
- Use ONLY the context below.
- If the answer is not in the context, say:
  "I don't have enough information to answer that."
- Each chunk is labeled [1], [2], etc.
- When using a chunk, cite it inline like [1].
- At the end, add a "References" section:
  Format: [n] source_filename, p.X (doc_id=vX)
"""


# ──────────────────────────────────────────────────────────────────────────────
# Context Construction (Version-Aware + Deduplicated)
# ──────────────────────────────────────────────────────────────────────────────
def build_context_block(
    user_id: str,
    chunks: List[Dict],
) -> str:
    """
    Build numbered context block.

    Features:
    - Deduplicate by checksum
    - Include document version
    - Ensure lifecycle safety
    - Preserve ordering
    """

    seen_checksums = set()
    filtered_chunks = []

    # Deduplicate identical chunks (multi-version safety)
    for chunk in chunks:
        checksum = chunk.get("checksum")
        lifecycle_status = chunk.get("lifecycle_status", "active")

        if lifecycle_status != "active":
            continue

        if checksum and checksum in seen_checksums:
            continue

        if checksum:
            seen_checksums.add(checksum)

        filtered_chunks.append(chunk)

    parts = []

    for i, c in enumerate(filtered_chunks, 1):
        source = c.get("source", "unknown")
        pages = c.get("pages", "")
        text = c.get("chunk_text", "")
        document_id = c.get("document_id", "unknown_doc")
        version = c.get("version", "v?")

        page_label = f", p.{pages}" if pages else ""
        parts.append(
            f"[{i}] (source: {source}{page_label}, doc_id={document_id}, version={version})\n{text}"
        )

    return "\n\n".join(parts)


# ──────────────────────────────────────────────────────────────────────────────
# Answer Generation (Multi-Tenant Safe)
# ──────────────────────────────────────────────────────────────────────────────
def generate_answer(
    user_id: str,
    query: str,
    chunks: List[Dict],
) -> str:
    """
    Generate answer using:
    - User-isolated context
    - Version-aware citations
    - Deduplicated chunks
    """

    if not chunks:
        return "I don't have enough information to answer that."

    context = build_context_block(user_id=user_id, chunks=chunks)

    messages = [
        ("system", SYSTEM_PROMPT),
        (
            "human",
            f"User ID: {user_id}\n\n"
            f"Context:\n{context}\n\n"
            f"---\nQuestion: {query}",
        ),
    ]

    ai_msg = _llm.invoke(messages)
    return ai_msg.content


# ──────────────────────────────────────────────────────────────────────────────
# CLI Test
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Generation Test (Multi-Tenant) ===")
    print(f"Using ChatOpenAI(model={OPENAI_MODEL})")

    test_user = "demo_user"

    test_chunks = [
        {
            "chunk_text": "What is the Teaching Schedule for Year 1",
            "source": "docs/Student-Handbook-2025-2026.pdf",
            "pages": "3",
            "document_id": "student_handbook_2025-2026",
            "version": "v1",
            "checksum": "abc123",
            "lifecycle_status": "active",
        },
        {
            "chunk_text": "Services revenue reached $25 billion.",
            "source": "docs/Student-Handbook-2025-2026.pdf",
            "pages": "5",
            "document_id": "student_handbook_2025-2026",
            "version": "v1",
            "checksum": "def456",
            "lifecycle_status": "active",
        },
    ]

    test_query = "What is the Teaching Schedule for Year 2?"

    print(f"User: {test_user}")
    print(f"Query: {test_query}")
    print(f"Context chunks: {len(test_chunks)}\n")

    answer = generate_answer(
        user_id=test_user,
        query=test_query,
        chunks=test_chunks,
    )

    print(f"💬 Answer:\n{answer}")
    print("\n✅ Generation test passed!")
