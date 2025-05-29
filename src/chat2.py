import os
import time
import chromadb
from dotenv import load_dotenv
import google.generativeai as genai
from langdetect import detect
import re

load_dotenv()



def llm_classify_intent(user_input):
    """
    Use Gemini to classify user input intent.
    Returns one of: greeting, identity, thanks, empty, question, unknown
    """
    # Trim and handle empty here
    if user_input.strip() == "":
        return "empty"

    prompt = f"""
Classify this user input into one of the categories:
greeting, identity, thanks, empty, question, or unknown.

User input: "{user_input}"

Answer with just the category name.
"""
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(
        contents=prompt,
        generation_config={"temperature": 0}
    )
    intent = response.text.strip().lower()

    # Defensive fallback in case model answers unexpectedly
    valid_intents = {"greeting", "identity", "thanks", "empty", "question", "unknown"}
    if intent not in valid_intents:
        return "unknown"
    return intent


def detect_and_translate_to_english(text):
    try:
        lang = detect(text)
    except:
        lang = "en"
    if lang == "en":
        return text, "en"
    model = genai.GenerativeModel("gemini-2.0-flash")
    translation = model.generate_content(f"Translate this to English:\n\n{text}")
    return translation.text.strip(), lang


def translate_answer(text, target_lang):
    if target_lang.lower() == "en":
        return text
    model = genai.GenerativeModel("gemini-2.0-flash")
    prompt = f"Translate this sentence directly into {target_lang}. No explanation, no prefix, just the translated sentence:\n\n{text}"
    response = model.generate_content(prompt)
    clean_text = response.text.strip()
    prefixes = [
        "Here is a translation to",
        "Här är en översättning till",
        "svenska:", "swedish:",
        f"translation to {target_lang.lower()}",
        f"translated to {target_lang.lower()}",
        f"{target_lang.lower()}:"
    ]
    for prefix in prefixes:
        if clean_text.lower().startswith(prefix.lower()):
            clean_text = clean_text[len(prefix):].strip(": \n")
    return clean_text


chroma_path = os.path.abspath("chroma_db")

genai.configure(api_key=os.getenv("API_KEY"))

final_prompt_template = """
You are an expert multilingual assistant named **DocuMate** tasked with answering user questions based solely on the provided context extracted from documents.

Context (with sources):
{context}

Instructions:
- Detect the language of the user’s question and respond in the same language.
- Use **only** the information in the context to answer the question.
- If the answer is not present in the retrieved content, clearly say: "I don't have enough information to answer that based on the current documents."
- If you use multiple sources, mention their chapters or pages as references (e.g., "see Chapter 13 and 14").
- Write your answer in well-structured paragraphs, in the same language as the question.
- Always stay in character as a helpful, technically knowledgeable assistant.
- Use a concise, clear, and approachable tone suited to the user's language and technical background.
- If the context lacks sufficient information, respond exactly with: "I don't know."
- Do not add any information or assumptions beyond the context.
  Do not use outside knowledge or invent any information.
- If the question is about your identity, respond with: "I am DocuMate, an expert assistant designed to answer questions based on the provided document context."
- If the user input is empty, respond with: "Please type a question."
- If the user input is not a question, respond with: "Please ask a question based on the provided context."
- If the user input is not something you are instructed to answer, respond with: "I don't know."
Question:
{question}

Answer:

"""

client_chroma = chromadb.PersistentClient(path=chroma_path)
collections = [col.name for col in client_chroma.list_collections()]

if "my_texts" not in collections:
    print("❌ Collection 'my_texts' not found in ChromaDB. Exiting.")
    exit(1)

collection = client_chroma.get_collection(name="my_texts")

count = collection.count()
if count == 0:
    print("❌ Warning: Collection 'my_texts' is empty. Please add documents first.")
    exit(1)


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

def process_user_input(user_input):
    """
    Full chat pipeline, returns final response string.
    """
    try:
        language = detect(user_input).split("-")[0].lower()
        if language != "en" and user_input.lower() in ["hi", "hello", "hey", "yo", "hej", "tjena"]:
            language = "en"
    except Exception:
        language = "en"

    intent = llm_classify_intent(user_input)

    if intent == "greeting":
        reply = "Hi there! How can I help you today?"
        return translate_answer(reply, language) if language != "en" else reply

    elif intent == "identity":
        reply = "I am DocuMate, an expert assistant designed to answer questions based on the provided document context."
        return translate_answer(reply, language) if language != "en" else reply

    elif intent == "thanks":
        reply = "You're welcome!"
        return translate_answer(reply, language) if language != "en" else reply

    elif intent == "empty":
        return "Please type a question."

    elif intent == "question":
        translated_input, user_lang = detect_and_translate_to_english(user_input)
        query_embedding = embed_query(translated_input)
        if query_embedding is None:
            return "Could not embed query, please try again."

        try:
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=5,
                include=["documents", "distances", "metadatas"]
            )

            filtered_chunks = [
                (doc, meta) for doc, score, meta in zip(
                    results["documents"][0], results["distances"][0], results["metadatas"][0]
                ) if score < 0.6
            ]

            context_chunks = []
            for doc, meta in filtered_chunks:
                chapter = meta.get("chapter", "Unknown Chapter")
                page = meta.get("page", "Unknown Page")
                context_chunks.append(f"[Chapter: {chapter}, Page: {page}]\n{doc.strip()}")
            context = "\n\n".join(context_chunks).strip()

            prompt = final_prompt_template.format(context=context, question=user_input)
            model = genai.GenerativeModel("gemini-2.0-flash")
            response = model.generate_content(
                contents=prompt,
                generation_config={"temperature": 0.2}
            )

            if "I don't know" in response.text:
                final_answer = "I don't know. Please ask a question based on the provided context."
            else:
                final_answer = translate_answer(response.text, user_lang)
            return final_answer

        except Exception as e:
            return f"❌ Error generating response: {e}"

    else:
        return "I don't know."

