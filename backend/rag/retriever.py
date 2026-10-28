"""
RAG Retriever — Queries the indexed knowledge base for relevant context.

This is the core interface that the LLM agents (Phase 5) will use.
Given a query about a network situation, it retrieves the most relevant
knowledge base passages to ground the agent's reasoning.

Usage:
    # As a module
    from rag.retriever import RAGRetriever
    rag = RAGRetriever()
    results = rag.query("URLLC latency exceeded 1ms, what should I do?")

    # Standalone test
    python -m rag.retriever "URLLC latency violation"
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Optional

import chromadb
from chromadb.utils import embedding_functions


DB_DIR = Path(__file__).parent / "chroma_db"
COLLECTION_NAME = "6g_knowledge"


class RAGRetriever:
    """
    Retrieves relevant knowledge from the 6G standards knowledge base.
    
    The retriever:
    1. Takes a natural language query (e.g., "URLLC latency is too high")
    2. Embeds the query using the same model used for indexing
    3. Finds the most similar knowledge base chunks via vector similarity
    4. Returns the text and metadata of the top matches
    
    The returned context is designed to be injected into an LLM prompt
    for grounded reasoning about network management decisions.
    """

    def __init__(self, use_local: bool = True):
        """
        Initialize the retriever.
        
        Args:
            use_local: If True, uses free local HuggingFace embeddings.
                       If False, uses OpenAI embeddings (requires OPENAI_API_KEY).
        """
        # Load env
        env_path = Path(__file__).parent.parent / ".env"
        if env_path.exists():
            try:
                from dotenv import load_dotenv
                load_dotenv(env_path)
            except ImportError:
                pass

        # Get embedding function (must match what was used for indexing)
        self.ef = self._get_embedding_function(use_local)

        # Connect to ChromaDB
        if not DB_DIR.exists():
            raise FileNotFoundError(
                f"RAG index not found at {DB_DIR}. "
                "Run: python -m rag.indexer --local"
            )

        self.client = chromadb.PersistentClient(path=str(DB_DIR))
        try:
            self.collection = self.client.get_collection(
                name=COLLECTION_NAME,
                embedding_function=self.ef,
            )
        except Exception:
            raise FileNotFoundError(
                f"Collection '{COLLECTION_NAME}' not found. "
                "Run: python -m rag.indexer --local"
            )

        self._chunk_count = self.collection.count()

    def _get_embedding_function(self, use_local: bool):
        """
        Must match whatever was used during indexing.
        Uses ChromaDB's default ONNX embeddings (no PyTorch dependency).
        """
        if not use_local:
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                return embedding_functions.OpenAIEmbeddingFunction(
                    api_key=api_key,
                    model_name="text-embedding-3-small"
                )
        return embedding_functions.DefaultEmbeddingFunction()

    def query(
        self,
        question: str,
        n_results: int = 5,
        min_relevance: float = 0.0,
    ) -> List[Dict]:
        """
        Query the knowledge base with a natural language question.
        
        Args:
            question: The query string (e.g., "URLLC latency exceeded target")
            n_results: Number of results to return (default 5)
            min_relevance: Minimum relevance score (0-1, 0 = return everything)
        
        Returns:
            List of dicts, each with:
                - text: The retrieved passage
                - source: Which document it came from
                - score: Relevance score (lower distance = more relevant)
                - chunk_index: Position within the source document
        """
        results = self.collection.query(
            query_texts=[question],
            n_results=min(n_results, self._chunk_count),
        )

        retrieved = []
        for i in range(len(results["documents"][0])):
            distance = results["distances"][0][i] if results["distances"] else 0
            metadata = results["metadatas"][0][i] if results["metadatas"] else {}

            # Convert distance to similarity score (ChromaDB uses L2 distance)
            similarity = 1.0 / (1.0 + distance)

            if similarity >= min_relevance:
                retrieved.append({
                    "text": results["documents"][0][i],
                    "source": metadata.get("source", "unknown"),
                    "score": round(similarity, 4),
                    "distance": round(distance, 4),
                    "chunk_index": metadata.get("chunk_index", -1),
                })

        return retrieved

    def get_context_for_prompt(
        self,
        question: str,
        n_results: int = 3,
        max_tokens: int = 2000,
    ) -> str:
        """
        Get a formatted context string ready to inject into an LLM prompt.
        
        Returns a clean text block with source attributions that can be
        directly pasted into a system/user prompt for the LLM agent.
        """
        results = self.query(question, n_results=n_results)

        if not results:
            return "No relevant knowledge found in the knowledge base."

        context_parts = []
        total_chars = 0

        for r in results:
            text = r["text"]
            # Rough token estimate: 1 token ≈ 4 characters
            if total_chars + len(text) > max_tokens * 4:
                break
            context_parts.append(
                f"[Source: {r['source']} | Relevance: {r['score']:.2f}]\n{text}"
            )
            total_chars += len(text)

        return "\n\n---\n\n".join(context_parts)

    def query_for_diagnosis(self, slice_name: str, issue_type: str, metrics: Dict) -> str:
        """
        Specialized query for network slice diagnosis.
        Builds a targeted query from the current network situation.
        
        Args:
            slice_name: "eMBB", "URLLC", or "mMTC"
            issue_type: "throughput_low", "latency_high", "queue_buildup", "sla_violation"
            metrics: Current slice metrics dict
        
        Returns:
            Formatted context string for the diagnosis agent.
        """
        # Build a natural language query from the situation
        query_parts = [f"{slice_name} slice"]

        if issue_type == "latency_high":
            query_parts.append(f"latency exceeded target at {metrics.get('latency_ms', '?')}ms")
            query_parts.append("diagnosis and remediation for high latency")
        elif issue_type == "throughput_low":
            query_parts.append(f"throughput dropped to {metrics.get('throughput_mbps', '?')} Mbps")
            query_parts.append("diagnosis and remediation for low throughput")
        elif issue_type == "sla_violation":
            query_parts.append("SLA violation detected")
            query_parts.append("root cause analysis and recommended actions")
        elif issue_type == "queue_buildup":
            query_parts.append(f"queue length at {metrics.get('queue_length', '?')}")
            query_parts.append("congestion diagnosis")
        else:
            query_parts.append(f"{issue_type} issue")

        query = " ".join(query_parts)
        return self.get_context_for_prompt(query, n_results=4)

    @property
    def status(self) -> Dict:
        """Return current index status."""
        return {
            "indexed_chunks": self._chunk_count,
            "db_path": str(DB_DIR),
            "collection": COLLECTION_NAME,
        }


# ── CLI Testing ──────────────────────────────────────────

def main():
    """Run interactive queries against the knowledge base."""
    query_text = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else None

    print("=" * 60)
    print("  6G Knowledge Base — RAG Retriever")
    print("=" * 60)

    try:
        rag = RAGRetriever(use_local=True)
        print(f"✅ Connected. {rag.status['indexed_chunks']} chunks indexed.\n")
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return

    if query_text:
        # Single query mode
        results = rag.query(query_text, n_results=3)
        print(f"Query: \"{query_text}\"\n")
        for i, r in enumerate(results):
            print(f"--- Result {i+1} (score: {r['score']:.3f}, source: {r['source']}) ---")
            print(r["text"][:300])
            print()
    else:
        # Interactive mode
        print("Type a question about 6G network management (or 'quit' to exit):\n")
        while True:
            try:
                q = input("❓ ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not q or q.lower() in ("quit", "exit", "q"):
                break

            results = rag.query(q, n_results=3)
            print()
            for i, r in enumerate(results):
                print(f"  [{r['source']} | score: {r['score']:.3f}]")
                preview = r["text"][:200].replace("\n", " ")
                print(f"  {preview}...\n")

            # Also show the formatted prompt context
            print("  === Context for LLM Prompt ===")
            context = rag.get_context_for_prompt(q, n_results=2)
            print(f"  {context[:500]}...\n")


if __name__ == "__main__":
    main()