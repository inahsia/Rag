# modules/vector_store.py

import faiss
import numpy as np
from typing import List, Tuple

class VectorStore:
    def __init__(self, dim: int):
        self.index = faiss.IndexFlatL2(dim)
        self.chunks = []  # Store original chunks for retrieval

    def add(self, embeddings: np.ndarray, texts: List[str]):
        self.index.add(embeddings)
        self.chunks.extend(texts)

    def search(self, query_embedding: np.ndarray, top_k: int = 3) -> List[Tuple[str, float]]:
        """
        Search for the most similar chunks.
        Returns: List of (chunk_text, similarity_score)
        """
        D, I = self.index.search(query_embedding, top_k)
        results = []
        for i, dist in zip(I[0], D[0]):
            results.append((self.chunks[i], dist))
        return results
