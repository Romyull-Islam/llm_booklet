"""
rag_agent_small.py: the section 5.7 capstone, wired the way section 4.2 says to
wire a 135M model.

`rag_agent.py` builds the capstone on `agent.py`, which uses the OpenAI `tools`
API. Section 4.2 states plainly that SmolLM2-135M cannot use that API, and it is
right: running rag_agent.py the model never calls search_knowledge_base
(steps taken: 0) and invents an answer instead. This version registers the same
retriever as a small_tools-style tool, which is the mechanism the booklet
actually recommends for models this size.
"""
import json
import os
import sys

from openai import OpenAI

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, "..", "week5_rag"))

from retriever import HybridRetriever                    # noqa: E402
from small_tools import TOOLS, build_prompt, run_small_agent   # noqa: E402

client = OpenAI(base_url=os.environ.get("LLM_BASE_URL", "http://localhost:8000/v1"),
                api_key="EMPTY")
MODEL = os.environ.get("LLM_MODEL", "HuggingFaceTB/SmolLM2-135M-Instruct")

records = json.load(open(os.path.join(HERE, "..", "data", "chunks.json")))
RETRIEVER = HybridRetriever(records)


def search_knowledge_base(query):
    hits = RETRIEVER.retrieve(query, top_k=3)
    if not hits:
        return "No relevant passages found."
    return " | ".join(f"[{i+1}] {h['text'][:200]}" for i, h in enumerate(hits))


# Register alongside the calculator and word_count the booklet already defines.
TOOLS["search_docs"] = {
    "args": ["query"],
    "doc": "search the documentation for a topic",
    "fn": search_knowledge_base,
}


def complete(prompt):
    r = client.chat.completions.create(
        model=MODEL, messages=[{"role": "user", "content": prompt}],
        temperature=0.0, max_tokens=96,
    )
    return r.choices[0].message.content or ""


for question in [
    "What is the maximum parameter count for any model in this booklet?",
    "What does the documentation say about rate limits?",
]:
    print(f"\n=== {question}")
    answer, transcript = run_small_agent(complete, question, max_steps=3)
    fired = transcript.count("\nTOOL: ")
    print(f"tool calls executed: {fired}")
    print("ANSWER:", answer[:300])
