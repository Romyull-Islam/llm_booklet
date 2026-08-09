# week5_rag/generate.py
from openai import OpenAI

import os

client = OpenAI(base_url=os.environ.get("LLM_BASE_URL", "http://localhost:8000/v1"),
                api_key="EMPTY")
MODEL = os.environ.get("LLM_MODEL", "HuggingFaceTB/SmolLM2-135M-Instruct")

SYSTEM = """You answer questions using only the numbered sources provided.

Rules:
1. Every factual claim must end with a citation marker like [1] or [2][3].
2. If the sources do not contain the answer, reply exactly: "The provided sources do not answer this question."
3. Never use knowledge outside the sources.
4. Be concise. Do not restate the question."""

def build_prompt(query, chunks):
    blocks = []
    for i, c in enumerate(chunks, start=1):
        header = c.get("title") or c.get("doc_id", "")
        blocks.append(f"[{i}] {header}\n{c['text']}")
    context = "\n\n---\n\n".join(blocks)
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"Sources:\n\n{context}\n\nQuestion: {query}"},
    ]

def answer(query, retriever, top_k=5):
    chunks = retriever.retrieve(query, top_k=top_k)
    resp = client.chat.completions.create(
        model=MODEL,
        messages=build_prompt(query, chunks),
        temperature=0.2,          # low: this is an extraction task, not a creative one
        max_tokens=600,
    )
    return {
        "answer": resp.choices[0].message.content,
        "sources": [{"n": i + 1, "title": c.get("title"), "url": c.get("url"),
                     "chunk_id": c["chunk_id"]} for i, c in enumerate(chunks)],
    }
