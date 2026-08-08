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
    """Search the documentation for passages relevant to a question.

    Pass the user's question verbatim, as a full sentence. Do NOT shorten it to
    keywords: the retriever is hybrid dense plus BM25 and scores whole questions
    far better than fragments. "What is the maximum parameter count?" works;
    "maximum parameter count" does not.

    Returns numbered passages with their source titles."""
    hits = RETRIEVER.retrieve(query, top_k=4)
    if not hits:
        return "No relevant passages found."
    return "\n\n".join(
        f"[{i+1}] {h.get('title','')} (chunk {h['chunk_id']})\n{h['text'][:800]}"
        for i, h in enumerate(hits)
    )

SYSTEM = """You are a documentation assistant with a search tool.

Rules:
- For any question about the documentation, call search_knowledge_base immediately.
  Do not announce that you are going to search. Just search.
- Answer only from the passages the tool returns. Cite them like [1] or [2].
- If the passages do not contain the answer, say exactly:
  "The documentation does not cover this." Do not guess, and do not build an answer
  out of a passage title.
"""

if __name__ == "__main__":
    import sys
    # A question this corpus can actually answer. Asking something the corpus does
    # not cover is a good test too, but make it a deliberate one, not the default.
    q = " ".join(sys.argv[1:]) or "What is the maximum parameter count for any model in this booklet?"
    # force_first_call=True: a documentation question cannot be answered without
    # searching, so do not leave that to the model's judgement.
    ans, trace = run_agent(q, system=SYSTEM, force_first_call=True)
    print("\nANSWER:", ans)
    print(f"\ntool calls: {sum(1 for m in trace if m.get('role') == 'tool')}")
