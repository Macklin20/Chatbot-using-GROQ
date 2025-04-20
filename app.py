import os
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# Load env variables
load_dotenv()

# Dark mode toggle
dark_mode = st.sidebar.checkbox("🌙 Dark Mode")

# Custom styles
st.markdown(f"""
    <style>
        body {{
            background-color: {"#0e1117" if dark_mode else "white"};
            color: {"white" if dark_mode else "black"};
        }}
        .chat-container {{
            display: flex;
            flex-direction: column-reverse;
        }}
        .chat-message {{
            padding: 10px 15px;
            border-radius: 10px;
            margin-bottom: 10px;
            width: fit-content;
            max-width: 85%;
            font-family: Arial, sans-serif;
            color: {"#fff" if dark_mode else "black"};
            background-color: {"#1a1a1a" if dark_mode else "#F1F0F0"};
        }}
        .chat-user {{
            background-color: {"#2f6e41" if dark_mode else "#DCF8C6"};
            align-self: flex-end;
            margin-left: auto;
        }}
        .chat-bot {{
            background-color: {"#1e1e1e" if dark_mode else "#F1F0F0"};
            align-self: flex-start;
            margin-right: auto;
        }}
        .chat-error {{
            background-color: #ffcccc;
            align-self: center;
            color: red;
        }}
    </style>
""", unsafe_allow_html=True)

# API Key
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    st.error("GROQ_API_KEY not found in .env")
    st.stop()

st.header("🤖 Groq AI Chatbot")

model_name = st.selectbox("Select Groq Model", ["llama3-8b-8192", "qwen-2.5-32b"])

# Initialize ChatGroq
try:
    chat = ChatGroq(
        temperature=0.7,
        groq_api_key=groq_api_key,
        model_name=model_name
    )
except Exception as e:
    st.error(f"Chat initialization failed: {str(e)}")
    st.stop()

# Chat state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Ask function
def ask_question():
    user_msg = st.session_state.user_input.strip()
    if user_msg:
        try:
            response = chat.invoke(user_msg)
            st.session_state.chat_history.insert(0, {"sender": "🤖 Groq", "msg": response.content})
            st.session_state.chat_history.insert(0, {"sender": "🧑 You", "msg": user_msg})
            st.session_state.user_input = ""
        except Exception as e:
            st.session_state.chat_history.insert(0, {"sender": "Error", "msg": str(e)})
    else:
        st.warning("Please enter a question.")

# Input box with on-change callback
st.text_input("You:", placeholder="Ask a question...", key="user_input", on_change=ask_question)

# Display chat history
st.subheader("Conversation:")
st.markdown('<div class="chat-container">', unsafe_allow_html=True)

# Loop through messages with controls
for i, item in enumerate(st.session_state.chat_history):
    sender, msg = item["sender"], item["msg"]
    css_class = "chat-user" if "You" in sender else "chat-bot"
    if sender == "Error":
        css_class = "chat-error"

    # Message HTML
    st.markdown(
        f'<div class="chat-message {css_class}"><strong>{sender}:</strong> {msg}</div>',
        unsafe_allow_html=True
    )

    # Only show buttons for Groq responses
    if sender == "🤖 Groq":
        col1, col2 = st.columns([1, 1])
        with col1:
            st.download_button(
                label="📥 Download",
                data=msg,
                file_name=f"groq_response_{i+1}.txt",
                mime="text/plain",
                key=f"download_{i}"
            )
        with col2:
            if st.button("🗑️ Clear", key=f"clear_{i}"):
                st.session_state.chat_history.pop(i)
                st.rerun()

st.markdown("</div>", unsafe_allow_html=True)

# Clear all
if st.button("🧹 Clear Entire Chat"):
    st.session_state.chat_history = []
    st.rerun()
