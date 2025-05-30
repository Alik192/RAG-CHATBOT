# DocuMate - RAG Chatbot

## Overview

DocuMate is a Retrieval-Augmented Generation (RAG) chatbot that answers questions based on a specific document. It uses ChromaDB for document storage and retrieval, and Google's Gemini model for generating responses.

## Project Structure


*   **`README.md`**: This file, providing an overview of the project.
*   **`chroma_db/`**: Contains the ChromaDB database files for storing and indexing document embeddings.
*   **`data/`**: Contains the PDF document used as the knowledge source for the chatbot.
*   **`src/`**: Contains the source code for the chatbot.
    *   **`chat.py`**: Implements the core chatbot logic, including document retrieval, and response generation.
    *   **`chatbot.py`**: Implements the Streamlit user interface for the chatbot.
    *   **`Evaluering.ipynb`**: A Jupyter Notebook used for evaluating the chatbot's performance.
    *   **`main.py`**: Loads the PDF, chunks it, and stores it in ChromaDB.
*   **`venv/`**: (Optional) A virtual environment containing the project's dependencies.
*   **`requirements.txt`**: Lists the project's Python dependencies.


## File Descriptions

Here's a brief description of each key file:

*   **`src/chat.py`**:
    *   **Purpose:** Contains the main logic for the chatbot.
    *   **Description:** This file handles user input, detects the intent, retrieves relevant documents from ChromaDB, and generates a response using the Gemini model.
*   **`src/chatbot.py`**:
    *   **Purpose:** Implements the Streamlit user interface.
    *   **Description:** This file creates a Streamlit app that allows users to interact with the chatbot through a chat interface. It handles user input, displays the chat history, and calls the `process_user_input` function from `chat.py` to generate responses.
*   **`Evaluering.ipynb`**:
    *   **Purpose:** Notebook for evaluating the chatbot's performance.
    *   **Description:** This Jupyter Notebook contains code for evaluating the chatbot's responses to a set of predefined questions. It compares the chatbot's answers to ideal answers and assigns scores based on relevance and correctness.
*   **`src/main.py`**:
    *   **Purpose:** Loads the PDF document, chunks it into smaller pieces, and stores the chunks in ChromaDB.
    *   **Description:** This script reads the PDF, splits the text into manageable chunks, generates embeddings for each chunk using the Gemini embedding model, and stores the embeddings and text in the ChromaDB database. This prepares the data for the chatbot to use.
*   **`data/`**:
    *   **Purpose:** Stores the PDF document used as the knowledge source.
    *   **Description:** This directory contains the PDF file that the chatbot uses to answer questions.
*   **`chroma_db/`**:
    *   **Purpose:** Stores the ChromaDB database.
    *   **Description:** This directory contains the files and subdirectories that make up the ChromaDB database, including the index files and the SQLite database file.
*   **`requirements.txt`**:
    *   **Purpose:** Lists the project's Python dependencies.
    *   **Description:** This file contains a list of all the Python packages required to run the project, along with their versions. It can be used to easily install the dependencies using `pip install -r requirements.txt`.
