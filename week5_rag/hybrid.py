# week5_rag/hybrid.py
import math, re
import numpy as np
from collections import Counter

def tokenize(s):
    return re.findall(r"[a-z0-9]+", s.lower())

class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.docs = [tokenize(d) for d in docs]
        self.k1, self.b = k1, b
        self.N = len(self.docs)
        self.avgdl = sum(len(d) for d in self.docs) / max(self.N, 1)
        self.tf = [Counter(d) for d in self.docs]
        df = Counter()
        for d in self.docs:
            df.update(set(d))
        self.idf = {t: math.log(1 + (self.N - n + 0.5) / (n + 0.5)) for t, n in df.items()}

    def scores(self, query):
        q = tokenize(query)
        out = np.zeros(self.N)
        for i, tf in enumerate(self.tf):
            dl = len(self.docs[i])
            s = 0.0
            for t in q:
                if t not in tf:
                    continue
                f = tf[t]
                s += self.idf.get(t, 0.0) * f * (self.k1 + 1) / (
                    f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
            out[i] = s
        return out

    def search(self, query, k=50):
        s = self.scores(query)
        idx = np.argsort(-s)[:k]
        return [(int(i), float(s[i])) for i in idx]

def rrf(rank_lists, k=60):
    """Reciprocal rank fusion. Combines rankings without needing comparable scores."""
    fused = {}
    for ranks in rank_lists:
        for pos, doc_id in enumerate(ranks):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + pos + 1)
    return sorted(fused, key=fused.get, reverse=True)
