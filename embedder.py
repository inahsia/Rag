# modules/embedder.py

from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np

# Load model only once
model = SentenceTransformer('all-MiniLM-L6-v2')

def embed_text(texts: List[str]) -> np.ndarray:
    """
    Converts list of text chunks into embeddings (vectors).
    """
    embeddings = model.encode(texts, show_progress_bar=True)
    return embeddings
