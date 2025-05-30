import os
import uuid
from dotenv import load_dotenv
from pypdf import PdfReader
from chromadb import PersistentClient
from chromadb.utils.embedding_functions import EmbeddingFunction
import google.generativeai as genai
from tqdm import tqdm
import time 
import os
import streamlit as st
# api key from streamlit secrets
API_KEY = st.secrets["API_KEY"]
#those were used when running locally
# --- Load environment ---
#load_dotenv()
#GEMINI_API_KEY = os.getenv("API_KEY") 

# --- Set up Gemini Embedding Function for Chroma ---
class GeminiEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input_texts: list[str]) -> list[list[float]]:
        #genai.configure(api_key=GEMINI_API_KEY)
        genai.configure(api_key=API_KEY)

        model = "models/embedding-001"

        embeddings = []
        for text in tqdm(input_texts, desc="🔄 Embedding via Gemini"):
            retry_count = 0
            while retry_count < 3:
                try:
                    response = genai.embed_content(
                        model=model,
                        content=text,
                        task_type="retrieval_document",
                        title="RAG chunk"
                    )
                    embeddings.append(response["embedding"])
                    time.sleep(0.5)  # Optional delay to avoid rate limits
                    break
                except Exception as e:
                    retry_count += 1
                    print(f"❌ Error embedding text: {e}, retrying ({retry_count}/3)...")
                    time.sleep(2)
            else:
                embeddings.append([0.0] * 768)  # Fallback in case of repeated failure
        return embeddings

# --- Load and chunk PDF ---
def chunk_text(text: str, chunk_size=3000, overlap=600) -> list[str]:
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i:i + chunk_size])
    print(f"📄 Total text chunks: {len(chunks)}")
    return chunks

reader = PdfReader("data/chattbot.pdf")
full_text = "".join(page.extract_text() for page in reader.pages)
chunks = chunk_text(full_text)

# --- Setup ChromaDB with Gemini embedding function ---
embedding_fn = GeminiEmbeddingFunction()
client = PersistentClient(path="chroma_db")
collection = client.get_or_create_collection(name="my_texts", embedding_function=embedding_fn)

# --- Create metadata and IDs ---
metadatas = [{"chunk_id": i, "source": "chattbot.pdf"} for i in range(len(chunks))]
ids = [str(uuid.uuid4()) for _ in chunks]

# --- Add to Chroma (embeddings generated automatically) ---
try:
    collection.add(documents=chunks, metadatas=metadatas, ids=ids)
    print("✅ Data stored in ChromaDB.")
    print(f"📦 Total records: {len(chunks)}")
except Exception as e:
    print(f"❌ Failed to store in ChromaDB: {e}")
