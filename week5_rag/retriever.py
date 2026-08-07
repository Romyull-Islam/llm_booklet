# week3_rag/retriever.py
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
        dense_ranks = [i for i, _ in self.vec.search(query, k_dense)]
        sparse_ranks = [i for i, _ in self.bm25.search(query, k_sparse)]
        fused = rrf([dense_ranks, sparse_ranks])
        candidates = [self.records[i] for i in fused[:30]]
        if self.reranker:
            return self.reranker.rerank(query, candidates, top_k)
        return candidates[:top_k]
