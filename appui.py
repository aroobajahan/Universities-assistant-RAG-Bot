import streamlit as st
from botrag_aroobajahan import ask_rag


st.set_page_config(page_title="RAG Chatbot", layout="centered")

st.title("RAG Chatbot")
st.write("Ask questions about universities in Pakistan")

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
prompt = st.chat_input("Ask something...")

if prompt:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Bot response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer, debug_info = ask_rag(prompt, with_debug=True)
            st.markdown(answer)

            if debug_info.get("candidates"):
                with st.expander("🔍 Retrieval scores (for tuning)"):
                    st.caption(f"Threshold: {debug_info['threshold']}")
                    for c in debug_info["candidates"]:
                        st.write(
                            f"**combined `{c['combined']}`** "
                            f"(dense `{c['dense']}`, keyword `{c['keyword']}`) — {c['question']}"
                        )

    st.session_state.messages.append({"role": "assistant", "content": answer})
