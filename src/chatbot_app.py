import sys
from pathlib import Path
from chatbot.chat_history import ChatHistory
from chatbot.llm_client import LLMClientFactory
import streamlit as st


# Set page config at the very beginning
st.set_page_config(page_title="Chatbot", page_icon="💬", initial_sidebar_state="collapsed")


def init_page(root_folder: Path):
    left_column, central_column, right_column = st.columns([2, 1, 2])

    with left_column:
        st.write(" ")

    with central_column:
        st.image(str(root_folder / "assets/bot-small.jpg"), width='stretch')
        st.markdown("""<h4 style='text-align: center; color: grey;'></h4>""", unsafe_allow_html=True)

    with right_column:
        st.write(" ")

    st.sidebar.title("Options")

@st.cache_resource
def init_welcome_message():
    with st.chat_message("assistant"):
        st.markdown("How can I help you today?")

def main(parameters):
    llm_client = LLMClientFactory.create_llm_client()

    init_page(Path(r'E:\rag-chat-bot'))
    init_welcome_message()

    st.session_state.messages = []
    chat_history = ChatHistory(total_length=10)

    # Supervise user input
    if user_input := st.chat_input("Input your question!"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(user_input)

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            streaming_response = llm_client.chat_completion(
                [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": user_input}
                ],
                stream=True
            )
            full_response = ""
            for token in streaming_response:
                full_response += token
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)

        # Add assistant response to chat history
        chat_history.append(f"question: {user_input}, answer: {full_response}")
        st.session_state.messages.append({"role": "assistant", "content": full_response})


# streamlit run chatbot_app.py
if __name__ == "__main__":
    try:
        main(None)
    except Exception as error:
        sys.exit(1)