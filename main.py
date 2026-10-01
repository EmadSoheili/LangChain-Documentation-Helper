from typing import Any, Dict, List
import streamlit as st
from backend.core import run_llm


def _format_sources(context_docs: List[Any]) -> List[str]:
    return list(set(                                                        # This is to make sure there is no duplicate sources being shown.
        str((meta.get("source") or "Unknown"))
        for doc in (context_docs or [])
        if (meta := (getattr(doc, "metadata", None) or {})) is not None
    ))


st.set_page_config(page_title="LangChain Documentation Helper", layout='centered')      # Configure basic setting of the application
st.title("LangChain Documentation Helper")      # Title to display

with st.sidebar:                                                    # Create a side bar
    st.subheader("Session")                                         # Header of the side bar
    if st.button("Clear chat", use_container_width=True):           # Create a button
        st.session_state.pop("messages", None)                      # Pop (clear) the session state message (the dictionarry we save all the intermediary messages in)
        st.rerun()                                                  # Rerun the app


if "messages" not in st.session_state:          # If there is no message in the dictionary, we will add an item to start with.
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Ask me anything about LangChain docs. I will retrieve relevant context and cite the source.",
            "source": [],            
        }
    ]


for msg in st.session_state.messages:           # Iterate over the messages
    with st.chat_message(msg["role"]):          # Show role (e.g. user or assistant)
        st.markdown(msg["content"])             # Show content from the session state (the dictionary)
        if msg.get("sources"):                  # Sources will be shown in a dropwon
            with st.expander("Sources"):
                for s in msg["sources"]:        # Loop through sources.
                    st.markdown(f"- {s}")       #Show each source and format it in markdown and as a list item because of the -


prompt = st.chat_input("Ask a question about LangChain.")           # Create input section with a placeholder text.
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt, "sources": []})
    with st.chat_message("user"):           # Showing user input
        st.markdown(prompt)                 # Format as markdown


    with st.chat_message("assistant"):          # AI reponse
        try:
            with st.spinner("Retrieving docs and generating answer..."):        # A spinner and the message will be shown as the response is loading.
                result: Dict[str, Any] = run_llm(prompt)                                        # Getting the rag response from llm using the prompt as the input.
                answer = str(result.get("answer", "")).strip() or "No answer returend!"         # Extracting answer from the RAG response.
                sources = _format_sources(result.get("context", []))                            # Get the sources

                st.markdown(answer)
                if sources:
                    with st.expander("Sources"):            # Print the sources like before.
                        for s in sources:
                            st.markdown(f"- {s}")           # Show as list.

                st.session_state.messages.append(
                    {"role": "assistant", "content": answer, "sources": sources}
                )



        except Exception as e:              # Error handling
            st.error("Failed to generate a response")
            st.exception(e)