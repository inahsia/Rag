# modules/data_loader.py

import fitz  # PyMuPDF
from typing import List
import re

def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)  # Collapse multiple spaces/newlines
    return text.strip()

def extract_text_from_pdf(file_path: str) -> str:
    doc = fitz.open(file_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    return clean_text(full_text)

def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> List[str]:
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i + chunk_size]
        chunks.append(' '.join(chunk))
        i += chunk_size - overlap  # Slide with overlap
    return chunks

def load_pdf_and_chunk(file_path: str, chunk_size=300, overlap=50) -> List[str]:
    raw_text = extract_text_from_pdf(file_path)
    return chunk_text(raw_text, chunk_size, overlap)
