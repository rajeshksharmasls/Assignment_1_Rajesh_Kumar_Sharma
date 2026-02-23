"""
RAG Classic — Multi-Tenant Entry Point.

Usage:

    # Start API server
    python main.py serve

    # Ingest a document
    python main.py ingest <user_id> <document_id> <file_path> [--version v2]

    Example:
    python main.py ingest user1 apple_q4 docs/Apple_Q24.pdf --version v1

    # Ask a question
    python main.py ask <user_id> "Question here" [--version v1] [--no-rerank] [--debug]

    Example:
    python main.py ask user1 "What was Apple's revenue?" --debug
"""

import sys
import os
import uvicorn


# ──────────────────────────────────────────────────────────────────────────────
# Server
# ──────────────────────────────────────────────────────────────────────────────


def serve():
    uvicorn.run("app.api:app", host="0.0.0.0", port=8000, reload=True)


# ──────────────────────────────────────────────────────────────────────────────
# Ingestion (User + Version Aware)
# ──────────────────────────────────────────────────────────────────────────────


def ingest(user_id: str, document_id: str, file_path: str, version: str):
    from code.ingestion import ingest_document
    from code.embedding import upsert_chunks, delete_previous_versions

    abs_path = os.path.abspath(file_path)

    if not os.path.exists(abs_path):
        print(f"❌ File not found: {abs_path}")
        sys.exit(1)

    print(f"\n📥 Ingesting for user={user_id}, doc={document_id}, version={version}")

    records = ingest_document(abs_path)

    # Delete old versions (clean update strategy)
    delete_previous_versions(user_id, document_id)

    upserted = upsert_chunks(
        user_id=user_id,
        document_id=document_id,
        records=records,
        version=version,
    )

    print(f"✅ Ingested {upserted} chunks")
    print("🎉 Done!")


# ──────────────────────────────────────────────────────────────────────────────
# Pretty Printing
# ──────────────────────────────────────────────────────────────────────────────


def _print_hits(label: str, hits):
    print(f"\n{'='*70}")
    print(f"  {label} ({len(hits)} results)")
    print(f"{'='*70}")
    for i, h in enumerate(hits, 1):
        pages = h.get("pages", "")
        page_label = f" | p.{pages}" if pages else ""
        doc = h.get("document_id", "")
        version = h.get("version", "")
        print(
            f"  [{i}] score={h['score']:.4f} | "
            f"{h['source']}{page_label} | "
            f"doc={doc} v={version}"
        )
        preview = h["chunk_text"][:200].replace("\n", " ")
        print(f"      {preview}...")
    print()


# ──────────────────────────────────────────────────────────────────────────────
# Ask (Full Multi-Tenant RAG Pipeline)
# ──────────────────────────────────────────────────────────────────────────────


def ask(
    user_id: str,
    question: str,
    version: str = None,
    use_reranker: bool = True,
    debug: bool = False,
):

    from code.retrieval import search
    from code.reranker import rerank
    from code.generation import generate_answer

    print(f"\n👤 User: {user_id}")
    print(f"🔍 Question: {question}")

    # ── Step 1: Retrieve ──────────────────────────────────────────────
    retrieved = search(
        user_id=user_id,
        query=question,
        version=version,
    )

    if debug or not use_reranker:
        _print_hits("📡 Retrieved (vector search)", retrieved)

    # ── Step 2: Rerank (optional) ──────────────────────────────────────
    if use_reranker:
        reranked = rerank(
            user_id=user_id,
            query=question,
            version=version,
        )

        if debug:
            _print_hits("🔀 Reranked", reranked)

            print("📊 Rerank impact:")
            retrieved_ids = [h["id"] for h in retrieved]
            for i, h in enumerate(reranked, 1):
                old_pos = (
                    retrieved_ids.index(h["id"]) + 1
                    if h["id"] in retrieved_ids
                    else "new"
                )
                print(f"    [{i}] {h['source']}: {old_pos} → {i}")
            print()

        chunks = reranked
    else:
        chunks = retrieved

    if not chunks:
        print("❌ No relevant results found.")
        return

    # ── Step 3: Generate ──────────────────────────────────────────────
    answer = generate_answer(
        user_id=user_id,
        query=question,
        chunks=chunks,
    )

    print(f"\n💬 Answer:\n{answer}\n")

    # ── Source Summary ────────────────────────────────────────────────
    print("📄 Sources used:")
    for i, c in enumerate(chunks, 1):
        pages = c.get("pages", "")
        page_label = f", p.{pages}" if pages else ""
        doc = c.get("document_id", "")
        version = c.get("version", "")
        print(
            f"  [{i}] {c['source']}{page_label} "
            f"(doc={doc}, v={version}, score={c['score']:.4f})"
        )


# ──────────────────────────────────────────────────────────────────────────────
# CLI Dispatcher
# ──────────────────────────────────────────────────────────────────────────────


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)

    command = sys.argv[1].lower()

    if command == "serve":
        serve()

    elif command == "ingest":
        if len(sys.argv) < 5:
            print(
                "Usage: python main.py ingest <user_id> <document_id> <file_path> [--version v1]"
            )
            sys.exit(1)

        user_id = sys.argv[2]
        document_id = sys.argv[3]
        file_path = sys.argv[4]

        version = "v1"
        if "--version" in sys.argv:
            idx = sys.argv.index("--version")
            if idx + 1 < len(sys.argv):
                version = sys.argv[idx + 1]

        ingest(user_id, document_id, file_path, version)

    elif command == "ask":
        if len(sys.argv) < 4:
            print(
                'Usage: python main.py ask <user_id> "<question>" [--version v1] [--no-rerank] [--debug]'
            )
            sys.exit(1)

        user_id = sys.argv[2]
        args = sys.argv[3:]

        use_reranker = True
        debug = False
        version = None
        question_parts = []

        i = 0
        while i < len(args):
            if args[i] == "--no-rerank":
                use_reranker = False
            elif args[i] == "--debug":
                debug = True
            elif args[i] == "--version" and i + 1 < len(args):
                version = args[i + 1]
                i += 1
            else:
                question_parts.append(args[i])
            i += 1

        if not question_parts:
            print("Question required.")
            sys.exit(1)

        question = " ".join(question_parts)

        ask(
            user_id=user_id,
            question=question,
            version=version,
            use_reranker=use_reranker,
            debug=debug,
        )

    else:
        print(f"Unknown command: {command}")
        print("Available commands: serve, ingest, ask")
        sys.exit(1)


if __name__ == "__main__":
    main()
