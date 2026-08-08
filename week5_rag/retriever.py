# week5_rag/retriever.py
from index import VectorIndex
from hybrid import BM25, rrf
from rerank import Reranker

class HybridRetriever:
    def __init__(self, records, use_reranker=True):
        self.records = records
        self.vec = VectorIndex().build(records)
        self.bm25 = BM25([r["text"] for r in records])
        self.reranker = Reranker() if use_reranker else None

    def retrieve(self, query, k_dense=50, k_sparse=50, top_k=5):
        # k_dense=0 or k_sparse=0 disables that retriever, which is what makes the
        # section 5.6 ablation an ablation rather than four identical rows.
        rank_lists = []
        if k_dense:
            rank_lists.append([i for i, _ in self.vec.search(query, k_dense)])
        if k_sparse:
            rank_lists.append([i for i, _ in self.bm25.search(query, k_sparse)])
        fused = rrf(rank_lists)
        candidates = [self.records[i] for i in fused[:30]]
        if self.reranker:
            return self.reranker.rerank(query, candidates, top_k)
        return candidates[:top_k]
