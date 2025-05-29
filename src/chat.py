import os
import sys

try:
    import pysqlite3
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    raise ImportError("pysqlite3 is required to run ChromaDB with the required SQLite version.")

import chromadb
#from dotenv import load_dotenv
import google.generativeai as genai

import streamlit as st

API_KEY = st.secrets["API_KEY"]

#load_dotenv()

# Configure Gemini
#genai.configure(api_key=os.getenv("API_KEY"))
genai.configure(api_key=API_KEY)

# Setup prompt template
final_prompt_template = """
You are a helpful assistant named **DocuMate**. You MUST follow the rules below:

RULES:
1. If the user input is a greeting (e.g., "hello", "hi", "hey", "good morning"), respond with a friendly greeting such as:
   "Hi there! How can I help you today?"

2. If the user says "thank you", "thanks", or shows gratitude, respond politely with:
   "You're welcome!" or "Anytime! Let me know if you need anything else."

3. If the input is NOT in English, respond with:
   "I'm sorry, I can only assist in English at the moment"
   Do not attempt to translate or answer.

4. If the user asks about your name, who you are, or your identity, respond with:
   "Hi there, I am DocuMate — an assistant designed to answer your questions based on the documents provided to me."

5. If the input is unrelated to a question or unclear, do not assume. Stay on script.

Context:
{context}

Instructions:
- Use only the context to answer the question below.
- If the answer is not in the context, say: "I don't have enough information to answer that based on the current documents."
- Do not make up information or go beyond the provided content.

Question:
{question}

Answer:
"""


# Initialize ChromaDB collection (assume it already exists and has data)
client_chroma = chromadb.PersistentClient(path=os.path.abspath("chroma_db"))
collection = client_chroma.get_collection(name="my_texts")

# Embed the query
def embed_query(query, model="models/embedding-001"):
    try:
        response = genai.embed_content(
            model=model,
            content=query,
            task_type="retrieval_query"
        )
        return response["embedding"]
    except Exception as e:
        print("❌ Error embedding query:", e)
        return None

# Main user input processing

def process_user_input(user_input):
    if not user_input.strip():
        return "Please type a question."

    query_embedding = embed_query(user_input)
    if query_embedding is None:
        return "Could not process your question. Try again."

    try:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=5,
            include=["documents", "distances"]
        )

        # Filter based on distance threshold
        context_chunks = [
            doc.strip() for doc, score in zip(
                results["documents"][0], results["distances"][0]
            ) if score < 0.6
        ]

        context = "\n\n".join(context_chunks).strip()

        # Instead of returning early, set context to a placeholder
        if not context:
            context = "No context available."

        prompt = final_prompt_template.format(context=context, question=user_input)
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(
            contents=prompt,
            generation_config={"temperature": 0.2}
        )

        return response.text.strip()

    except Exception as e:
        return f"❌ Error generating response: {e}"




