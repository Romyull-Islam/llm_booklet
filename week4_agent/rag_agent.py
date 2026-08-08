# week4_agent/rag_agent.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent / "week5_rag"))

import json
from retriever import HybridRetriever
from tools import tool, TOOLS
from agent import run_agent

records = json.load(open(Path(__file__).resolve().parent.parent / "data" / "chunks.json"))
RETRIEVER = HybridRetriever(records)

@tool
def search_knowledge_base(query: str) -> str:
    """Search the internal document collection for passages relevant to a question.
    Use this for any question about internal documentation, policies, or product details.
    Returns numbered passages with their source titles."""
    hits = RETRIEVER.retrieve(query, top_k=4)
    if not hits:
        return "No relevant passages found."
    return "\n\n".join(
        f"[{i+1}] {h.get('title','')} (chunk {h['chunk_id']})\n{h['text'][:800]}"
        for i, h in enumerate(hits)
    )

SYSTEM = """You are a documentation assistant.

Rules:
- For any question about internal documents, call search_knowledge_base FIRST.
- Cite the passage numbers you used, like [1] or [2].
- If the search returns nothing relevant, say so. Do not answer from memory.
- Use the calculator for arithmetic."""

if __name__ == "__main__":
    ans, trace = run_agent("What does our documentation say about rate limits?", system=SYSTEM)
    print("\nANSWER:", ans)
    print(f"\nsteps taken: {sum(1 for m in trace if m['role'] == 'tool')}")
