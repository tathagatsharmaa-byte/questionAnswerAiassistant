"""Streamlit interface for the AI question and answering application."""

import streamlit as st

from app.config.settings import ConfigurationError, get_settings
from app.services.gemini_service import GeminiService, GeminiServiceError
from app.utils.logging_config import configure_logging


def main() -> None:
    """Render the chat interface and handle submitted questions."""
    configure_logging()
    st.set_page_config(page_title="AI Question & Answering System", page_icon="💬")
    st.title("AI Question & Answering System")
    st.write("Ask questions and get helpful answers powered by Google Gemini.")

    try:
        settings = get_settings()
    except ConfigurationError as error:
        st.error(str(error))
        st.stop()
        return

    service = GeminiService(settings)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    with st.sidebar:
        st.subheader("Conversation")
        st.caption(f"Model: `{settings.model}`")
        st.caption(f"Temperature: `{settings.temperature}`")
        if st.button("Clear Conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input("Ask a question")
    if question is None:
        return

    if not question.strip():
        st.warning("Enter a question before submitting.")
        return

    user_message = {"role": "user", "content": question.strip()}
    st.session_state.messages.append(user_message)
    with st.chat_message("user"):
        st.markdown(user_message["content"])

    try:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = service.generate_answer(st.session_state.messages)
            st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
    except GeminiServiceError as error:
        st.error(str(error))


if __name__ == "__main__":
    main()
