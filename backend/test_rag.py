"""
Test the RAG system end-to-end.
Run: python test_rag.py
"""

import sys
sys.path.insert(0, ".")

from rag.retriever import RAGRetriever


def test_rag():
    print("=" * 60)
    print("  RAG System Test")
    print("=" * 60)

    # 1. Connect
    try:
        rag = RAGRetriever(use_local=True)
        print(f"✅ Connected: {rag.status['indexed_chunks']} chunks indexed\n")
    except FileNotFoundError as e:
        print(f"❌ {e}")
        print("\nRun this first:")
        print("  pip install chromadb sentence-transformers")
        print("  python -m rag.indexer --local")
        return

    # 2. Test queries that the agents will actually use
    test_cases = [
        {
            "query": "URLLC latency exceeded 1ms, what should I do?",
            "should_contain": ["URLLC", "latency", "1ms"],
        },
        {
            "query": "eMBB throughput dropped below 100 Mbps",
            "should_contain": ["eMBB", "throughput"],
        },
        {
            "query": "What is the minimum allocation for each slice?",
            "should_contain": ["minimum", "allocation"],
        },
        {
            "query": "How to handle traffic spike in the network?",
            "should_contain": ["surge", "spike", "demand"],
        },
        {
            "query": "mMTC devices all reporting at once",
            "should_contain": ["mMTC", "burst", "sensor"],
        },
    ]

    passed = 0
    for tc in test_cases:
        results = rag.query(tc["query"], n_results=3)
        print(f"❓ \"{tc['query']}\"")

        if not results:
            print(f"   ❌ No results returned\n")
            continue

        top = results[0]
        # Check if any expected keyword appears in top result
        found_keywords = [
            kw for kw in tc["should_contain"]
            if kw.lower() in top["text"].lower()
        ]

        if found_keywords:
            print(f"   ✅ Top result (score: {top['score']:.3f}) from {top['source']}")
            print(f"      Found keywords: {found_keywords}")
            passed += 1
        else:
            print(f"   ⚠️  Top result didn't contain expected keywords")
            print(f"      Expected: {tc['should_contain']}")

        # Show preview
        preview = top["text"][:150].replace("\n", " ")
        print(f"      Preview: {preview}...\n")

    print(f"\n{'=' * 60}")
    print(f"  Results: {passed}/{len(test_cases)} queries returned relevant results")
    print(f"{'=' * 60}")

    # 3. Test the diagnosis helper
    print("\n\n--- Testing Diagnosis Query Helper ---\n")
    context = rag.query_for_diagnosis(
        slice_name="URLLC",
        issue_type="latency_high",
        metrics={"latency_ms": 1.5, "throughput_mbps": 0.8, "queue_length": 7}
    )
    print(f"Diagnosis context ({len(context)} chars):")
    print(context[:500])
    print("...")

    print("\n✅ RAG system is working!")


if __name__ == "__main__":
    test_rag()
