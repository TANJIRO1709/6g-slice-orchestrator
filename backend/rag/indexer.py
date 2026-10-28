"""
RAG Indexer — Embeds knowledge base documents and stores them in ChromaDB.

Usage:
    python -m rag.indexer              # Index all docs using OpenAI embeddings
    python -m rag.indexer --local      # Index using free local HuggingFace embeddings
    python -m rag.indexer --status     # Check what's indexed
"""

import os
import argparse
from pathlib import Path
from typing import List

# ChromaDB for vector storage
import chromadb
from chromadb.utils import embedding_functions


# ── Config ───────────────────────────────────────────────

KB_DIR = Path(__file__).parent / "knowledge_base"
DB_DIR = Path(__file__).parent / "chroma_db"
COLLECTION_NAME = "6g_knowledge"
CHUNK_SIZE = 500        # characters per chunk
CHUNK_OVERLAP = 100     # overlap between chunks


def chunk_document(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Split a document into overlapping chunks.
    Each chunk is roughly chunk_size characters with overlap.
    Splits at paragraph boundaries when possible.
    """
    paragraphs = text.split("\n\n")
    chunks = []
    current_chunk = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        # If adding this paragraph exceeds chunk size, save current and start new
        if len(current_chunk) + len(para) > chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            # Keep overlap from end of previous chunk
            if overlap > 0 and len(current_chunk) > overlap:
                current_chunk = current_chunk[-overlap:] + "\n\n" + para
            else:
                current_chunk = para
        else:
            current_chunk = current_chunk + "\n\n" + para if current_chunk else para

    # Don't forget the last chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


def get_embedding_function(use_local: bool = False):
    """
    Get the embedding function.

    Priority:
    1. If use_local=True -> ChromaDB default (ONNX-based, no PyTorch needed)
    2. If OPENAI_API_KEY set and use_local=False -> OpenAI embeddings
    3. Fallback -> ChromaDB default
    """
    if not use_local:
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            print("☁️  Using OpenAI embeddings (text-embedding-3-small)")
            return embedding_functions.OpenAIEmbeddingFunction(
                api_key=api_key,
                model_name="text-embedding-3-small"
            )
        print("⚠️  OPENAI_API_KEY not set. Using local embeddings instead.")

    # ChromaDB's built-in default: all-MiniLM-L6-v2 via ONNX runtime.
    # No PyTorch, no sentence-transformers needed.
    print("📦 Using ChromaDB default embeddings (ONNX, no PyTorch required)")
    print("   First run downloads a small model (~80MB)...")
    return embedding_functions.DefaultEmbeddingFunction()


def index_knowledge_base(use_local: bool = False):
    """Read all knowledge base docs, chunk them, embed, and store in ChromaDB."""
    print("=" * 60)
    print("  6G Knowledge Base — RAG Indexer")
    print("=" * 60)

    # Load environment variables
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        from dotenv import load_dotenv
        load_dotenv(env_path)

    # Get embedding function
    ef = get_embedding_function(use_local)

    # Initialize ChromaDB
    DB_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(DB_DIR))

    # Delete existing collection if it exists (re-index from scratch)
    try:
        client.delete_collection(COLLECTION_NAME)
        print("🗑️  Cleared existing index")
    except Exception:
        pass

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"description": "6G network slicing knowledge base"}
    )

    # Read and chunk all markdown files
    md_files = sorted(KB_DIR.glob("*.md"))
    if not md_files:
        print(f"❌ No .md files found in {KB_DIR}")
        return

    all_chunks = []
    all_ids = []
    all_metadata = []

    for filepath in md_files:
        filename = filepath.name
        text = filepath.read_text(encoding="utf-8")
        chunks = chunk_document(text)

        print(f"\n📄 {filename}")
        print(f"   {len(text)} chars → {len(chunks)} chunks")

        for i, chunk in enumerate(chunks):
            chunk_id = f"{filepath.stem}_chunk_{i:03d}"
            all_chunks.append(chunk)
            all_ids.append(chunk_id)
            all_metadata.append({
                "source": filename,
                "chunk_index": i,
                "total_chunks": len(chunks),
            })

    # Add all chunks to ChromaDB
    print(f"\n🔄 Embedding {len(all_chunks)} chunks...")

    # ChromaDB has a batch limit, process in batches of 40
    batch_size = 40
    for start in range(0, len(all_chunks), batch_size):
        end = min(start + batch_size, len(all_chunks))
        collection.add(
            documents=all_chunks[start:end],
            ids=all_ids[start:end],
            metadatas=all_metadata[start:end],
        )

    print(f"\n✅ Indexed {len(all_chunks)} chunks from {len(md_files)} documents")
    print(f"   Stored in: {DB_DIR}")
    print(f"   Collection: {COLLECTION_NAME}")


def check_status():
    """Check what's currently indexed."""
    print("=" * 60)
    print("  RAG Index Status")
    print("=" * 60)

    if not DB_DIR.exists():
        print("❌ No index found. Run: python -m rag.indexer")
        return

    client = chromadb.PersistentClient(path=str(DB_DIR))
    try:
        collection = client.get_collection(COLLECTION_NAME)
        count = collection.count()
        print(f"✅ Collection '{COLLECTION_NAME}' exists")
        print(f"   Total chunks indexed: {count}")

        # Show a sample
        if count > 0:
            sample = collection.peek(limit=3)
            print(f"\n   Sample chunks:")
            for i, doc in enumerate(sample["documents"]):
                source = sample["metadatas"][i]["source"]
                preview = doc[:100].replace("\n", " ")
                print(f"   [{source}] {preview}...")
    except Exception:
        print("❌ Collection not found. Run: python -m rag.indexer")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Index 6G knowledge base for RAG")
    parser.add_argument("--local", action="store_true", help="Use free local embeddings instead of OpenAI")
    parser.add_argument("--status", action="store_true", help="Check index status")
    args = parser.parse_args()

    if args.status:
        check_status()
    else:
        index_knowledge_base(use_local=args.local)