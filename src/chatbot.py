import streamlit as st
from chat import process_user_input

# Configure the page
st.set_page_config(
    page_title="DocuMate",
    page_icon="🤖", 
    layout="centered"
)

# Handle reset trigger
if "reset_triggered" in st.session_state:
    del st.session_state["reset_triggered"]
    st.session_state.clear()
    st.query_params.update()

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Sidebar: About and Download
with st.sidebar:
    
    st.markdown("## 🧠 About DocuMate")
    st.write("""
    DocuMate is your intelligent assistant for document-based answers.

    **Current document:**
    *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow*  
    by Aurélien Géron
    """)
    
    if st.button("🔄 Reset Chat"):
        st.session_state["reset_triggered"] = True

    if st.session_state.chat_history:
        conversation_text = ""
        for sender, msg in st.session_state.chat_history:
            prefix = "User: " if sender == "user" else "DocuMate: "
            conversation_text += prefix + msg + "\n\n"

        st.download_button(
            label="📥 Download Conversation",
            data=conversation_text,
            file_name="documate_conversation.txt",
            mime="text/plain"
        )

# Main Title
st.markdown("""
<h1 style='text-align: center; color: #4CAF50;'>🤖 DocuMate</h1>
<p style='text-align: center;'>Your intelligent assistant for document-based questions.</p>
""", unsafe_allow_html=True)

# Welcome / How-to Message
st.success("👋 Ask anything from the book. DocuMate is here to help!")

# Optional Instructions Panel
with st.expander("📘 How to use DocuMate"):
    st.markdown("""
    - 💬 Ask any question related to the current document.
    - 📥 Download the conversation anytime using the sidebar.
    - 🔄 Use the **Reset Chat** button to start a new session.
    """)

# Chat input
user_input = st.chat_input("Type your question...")

if user_input:
    with st.spinner("DocuMate is thinking..."):
        response = process_user_input(user_input)
    st.session_state.chat_history.append(("user", user_input))
    st.session_state.chat_history.append(("bot", response))

# Display chat history
for sender, msg in st.session_state.chat_history:
    if sender == "user":
        st.chat_message("user").markdown(msg)
    else:
        st.chat_message("assistant").markdown(f"🤖: {msg}")
