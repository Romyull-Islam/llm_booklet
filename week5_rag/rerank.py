# week3_rag/rerank.py
from sentence_transformers import CrossEncoder

class Reranker:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L6-v2", device="cuda"):
        self.model = CrossEncoder(model_name, device=device, max_length=512)

    def rerank(self, query, candidates, top_k=5):
        """candidates: list of chunk records. Returns the top_k, best first."""
        pairs = [(query, c["text"]) for c in candidates]
        scores = self.model.predict(pairs, batch_size=16)
        ranked = sorted(zip(candidates, scores), key=lambda x: -x[1])[:top_k]
        return [{**c, "rerank_score": float(s)} for c, s in ranked]
