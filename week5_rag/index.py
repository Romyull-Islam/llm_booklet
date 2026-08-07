# week3_rag/index.py
import numpy as np
from sentence_transformers import SentenceTransformer

EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# all-MiniLM-L6-v2 is symmetric: no query prefix. Instruction-prefixed
# embedders (Qwen3-Embedding, e5, bge) need one; MiniLM does not.
QUERY_INSTRUCTION = ""

class VectorIndex:
    def __init__(self, model_name=EMB_MODEL, device="cuda"):
        self.model = SentenceTransformer(model_name, device=device)
        self.matrix = None
        self.records = []

    def build(self, records, batch_size=32):
        self.records = records
        texts = [r["text"] for r in records]
        self.matrix = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,      # so a dot product is cosine similarity
            show_progress_bar=True,
            convert_to_numpy=True,
        ).astype("float32")
        return self

    def search(self, query, k=50):
        q = self.model.encode(
            QUERY_INSTRUCTION + query,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype("float32")
        scores = self.matrix @ q
        idx = np.argsort(-scores)[:k]
        return [(int(i), float(scores[i])) for i in idx]

    def save(self, path):
        np.save(path + ".npy", self.matrix)

    def load(self, path, records):
        self.matrix = np.load(path + ".npy")
        self.records = records
        return self
