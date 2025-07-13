import numpy as np
from data_loader import load_pdf_and_chunk
from embedder import embed_text
from vector_store import VectorStore
from llm_interface import generate_answer

# Step 1: Load PDF and chunk it
chunks = load_pdf_and_chunk(r"C:\Users\singh\OneDrive\Desktop\Rag project\documents\1.pdf")

# Step 2: Embed the chunks
embeddings = embed_text(chunks)

# Step 3: Create vector store and add data
vector_store = VectorStore(dim=embeddings[0].shape[0])
vector_store.add(embeddings, chunks)

# Step 4: Embed user query
query = "How does fog computing help in smart traffic?"
query_embedding = embed_text([query])

# Step 5: Retrieve top 3 most relevant chunks
results = vector_store.search(query_embedding, top_k=3)
top_chunks = [chunk for chunk, score in results]

# Step 6: Ask Gemini to answer using top chunks
final_answer = generate_answer(query, top_chunks)

# Step 7: Print result
print("\n🎯 Final Answer from Gemini:\n")
print(final_answer)
