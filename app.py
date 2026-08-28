import streamlit as st

from src.rag import ask
from src.memory import ConversationMemory


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Stella",
    page_icon="✦",
    layout="centered",
)


# --------------------------------------------------
# Session state
# --------------------------------------------------

if "memory" not in st.session_state:
    st.session_state.memory = ConversationMemory()

if "messages" not in st.session_state:
    st.session_state.messages = []


# --------------------------------------------------
# Header
# --------------------------------------------------

st.title("✦ Stella")
st.caption("Your local RAG assistant")


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

with st.sidebar:

    st.header("Stella")

    st.write(
        "A local AI assistant powered by "
        "Ollama, ChromaDB, and RAG."
    )

    st.divider()

    if st.button("New conversation", use_container_width=True):

        st.session_state.memory = ConversationMemory()
        st.session_state.messages = []

        st.rerun()


# --------------------------------------------------
# Display conversation
# --------------------------------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


# --------------------------------------------------
# Chat input
# --------------------------------------------------

question = st.chat_input("Talk to Stella...")


if question:

    # Display user message immediately
    with st.chat_message("user"):
        st.markdown(question)

    # Save user message for UI
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    # Generate Stella response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer, results, search_query = ask(question, st.session_state.memory)

        st.markdown(answer)

        if results:
            with st.expander("Sources"):
                st.caption(f"Search query: {search_query}")
                for r in results:
                    st.markdown(f"- **{r['source']}** (page {r['page']}) — distance {r['distance']:.4f}")

    st.session_state.messages.append({"role": "assistant", "content": answer})

    # Save assistant response for UI
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
        }
    )