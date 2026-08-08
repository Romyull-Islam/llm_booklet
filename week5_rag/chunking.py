# week5_rag/chunking.py
import re

def chunk_text(text, chunk_chars=1200, overlap=200):
    """Split on paragraph boundaries, pack into ~chunk_chars windows with overlap."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks, cur = [], ""
    for p in paras:
        if len(cur) + len(p) + 2 <= chunk_chars:
            cur = f"{cur}\n\n{p}" if cur else p
        else:
            if cur:
                chunks.append(cur)
            cur = (cur[-overlap:] + "\n\n" + p) if (overlap and cur) else p
            while len(cur) > chunk_chars:
                chunks.append(cur[:chunk_chars])
                cur = cur[chunk_chars - overlap:]
    if cur:
        chunks.append(cur)
    return chunks

def build_corpus(docs, chunk_chars=1200, overlap=200):
    """docs: list of {'id', 'title', 'text', 'url'}. Returns flat chunk records."""
    records = []
    for d in docs:
        for i, c in enumerate(chunk_text(d["text"], chunk_chars, overlap)):
            records.append({
                "chunk_id": f"{d['id']}::{i}",
                "doc_id": d["id"],
                "title": d.get("title", ""),
                "url": d.get("url", ""),
                "text": c,
            })
    return records
