# llm_interface.py

import google.generativeai as genai
import config  # runs genai.configure(api_key=...)

# Use the working model
model = genai.GenerativeModel(model_name="gemini-1.5-flash")

def generate_answer(query: str, context_chunks: list[str]) -> str:
    try:
        # Combine the top chunks into a single context
        context = "\n\n".join(context_chunks)
        prompt = f"""You are an AI assistant. Use the following context to answer the question.

Context:
{context}

Question: {query}

Answer:"""

        response = model.generate_content(prompt)
        return response.text.strip()

    except Exception as e:
        return f"[Error] {e}"
