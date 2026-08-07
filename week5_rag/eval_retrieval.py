# week3_rag/eval_retrieval.py
import json
import numpy as np

def recall_at_k(retrieved_ids, gold_ids, k):
    return len(set(retrieved_ids[:k]) & set(gold_ids)) / max(len(gold_ids), 1)

def mrr(retrieved_ids, gold_ids):
    """Mean reciprocal rank: how high up the first correct hit appears."""
    for i, d in enumerate(retrieved_ids):
        if d in gold_ids:
            return 1.0 / (i + 1)
    return 0.0

def ndcg_at_k(retrieved_ids, gold_ids, k):
    dcg = sum(1.0 / np.log2(i + 2) for i, d in enumerate(retrieved_ids[:k]) if d in gold_ids)
    ideal = sum(1.0 / np.log2(i + 2) for i in range(min(len(gold_ids), k)))
    return dcg / ideal if ideal > 0 else 0.0

def evaluate(retriever, eval_set, k=5):
    """eval_set: list of {'question': str, 'gold_chunk_ids': [str, ...]}"""
    rows = []
    for ex in eval_set:
        hits = retriever.retrieve(ex["question"], top_k=k)
        ids = [h["chunk_id"] for h in hits]
        rows.append({
            f"recall@{k}": recall_at_k(ids, ex["gold_chunk_ids"], k),
            "mrr": mrr(ids, ex["gold_chunk_ids"]),
            f"ndcg@{k}": ndcg_at_k(ids, ex["gold_chunk_ids"], k),
        })
    return {m: round(float(np.mean([r[m] for r in rows])), 4) for m in rows[0]}

if __name__ == "__main__":
    from retriever import HybridRetriever
    records = json.load(open("data/chunks.json"))
    eval_set = json.load(open("data/eval_questions.json"))

    # This ablation table is the actual deliverable of Week 3.
    configs = {
        "dense only":            dict(k_sparse=0, use_reranker=False),
        "bm25 only":             dict(k_dense=0, use_reranker=False),
        "hybrid":                dict(use_reranker=False),
        "hybrid + reranker":     dict(use_reranker=True),
    }
    for name, cfg in configs.items():
        r = HybridRetriever(records, use_reranker=cfg.pop("use_reranker", True))
        print(f"{name:<22}", evaluate(r, eval_set, k=5))
