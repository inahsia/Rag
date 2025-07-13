# config.py
import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv(r'C:\Users\singh\OneDrive\Desktop\Rag project\.env')  # Load .env file

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("API key not found. Make sure .env is correctly loaded.")

genai.configure(api_key=api_key)
