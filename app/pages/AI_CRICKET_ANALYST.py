import streamlit as st
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from chatbot import ask_agent

st.set_page_config(page_title="AI Chat - IPL Intelligence", page_icon="🤖", layout="wide")

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []
if "chat_thread_id" not in st.session_state:
    st.session_state.chat_thread_id = "ai-chat-page-session"


# Title row - heading + clear button side by side

title_col, clear_col = st.columns([6, 1])
with title_col:
    st.title("🤖 AI Cricket Analyst")
with clear_col:
    st.write("")
    if st.button("🗑️ Clear", use_container_width=True):
        st.session_state.chat_messages = []
        st.rerun()

EXAMPLE_QUESTIONS = [
    "What is Kohli's strike rate?",
    "How many wickets does Bumrah have?",
    "MI vs CSK, who will win?",
    "Winner of IPL 2020?",
    "Average score at Wankhede?",
    "Highest run scorer?",
    "Rohit vs Virat, who is better?",
    "Tell me about Kohli's 2016 IPL performance",
    "Best all-time IPL XI",
]

pending_question = None


# Chat history render

for msg in st.session_state.chat_messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        

# Quick-question chips (only shown before conversation starts)
if len(st.session_state.chat_messages) == 0:
    st.write("Try one of these, or type your own question below:")
    chip_cols = st.columns(3)
    for i, q in enumerate(EXAMPLE_QUESTIONS):
        with chip_cols[i % 3]:
            if st.button(q, key="chip_" + str(i), use_container_width=True):
                pending_question = q
    st.write("")


# Chat input - kept at top level so Streamlit pins it to the bottom

typed_input = st.chat_input("Type your question here...")
user_input = pending_question or typed_input

if user_input:
    st.session_state.chat_messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing IPL data..."):
            answer = ask_agent(user_input, st.session_state.chat_thread_id)
            st.write(answer)
            

    st.session_state.chat_messages.append({"role": "assistant", "content": answer})

    if pending_question:
        st.rerun()