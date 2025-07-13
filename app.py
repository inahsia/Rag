# app.py

import streamlit as st
from data_loader import load_pdf_and_chunk
from embedder import embed_text
from vector_store import VectorStore
from llm_interface import generate_answer

st.set_page_config(page_title="PDF RAG Chatbot", layout="wide")

st.title("📄🤖 PDF RAG Chatbot with Gemini")
st.markdown("Upload a PDF and ask questions based on its content.")

uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file:
    # Save uploaded file locally
    with open(f"temp_uploaded.pdf", "wb") as f:
        f.write(uploaded_file.read())

    with st.spinner("Reading and chunking the PDF..."):
        chunks = load_pdf_and_chunk("temp_uploaded.pdf")

    with st.spinner("Embedding chunks..."):
        embeddings = embed_text(chunks)

    vector_store = VectorStore(dim=embeddings[0].shape[0])
    vector_store.add(embeddings, chunks)

    st.success("✅ PDF loaded! You can now ask questions.")

    query = st.text_input("💬 Ask your question")

    if query:
        query_embedding = embed_text([query])
        results = vector_store.search(query_embedding, top_k=3)
        top_chunks = [chunk for chunk, _ in results]

        with st.spinner("💡 Gemini is thinking..."):
            answer = generate_answer(query, top_chunks)

        st.subheader("🧠 Gemini's Answer:")
        st.write(answer)

        with st.expander("🔍 Top retrieved chunks (context)"):
            for i, (chunk, score) in enumerate(results):
                st.markdown(f"**Chunk {i+1} (Score: {score:.4f})**")
                st.markdown(chunk)
