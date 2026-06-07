import streamlit as st
import os
import rag_utils as ru

# Set page configuration
st.set_page_config(page_title="AI Document Analyst", page_icon="📄", layout="wide")

st.title("📄 AI Document Analyst")
st.markdown("Upload your PDF documents in the sidebar and ask questions. The AI will answer based **only** on the provided context!")

# Initialize chat history in session state so the chat doesn't disappear on refresh
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Sidebar for Document Upload ---
with st.sidebar:
    st.title("📂 Document Upload")
    st.write("Upload your PDFs here and click 'Process'.")
    
    # Allow multiple PDF uploads
    pdf_docs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True, type=["pdf"])
    
    if st.button("Process Documents", type="primary"):
        if not pdf_docs:
            st.warning("Please upload at least one PDF file first.")
        else:
            with st.spinner("Reading, chunking, and indexing documents..."):
                # 1. Extract text
                raw_text = ru.extract_text_from_pdf(pdf_docs)
                # 2. Split into chunks
                text_chunks = ru.get_text_chunks(raw_text)
                # 3. Create Vector Store
                ru.get_vector_store(text_chunks)
                st.success("✅ Processing Complete! You can now ask questions.")

# --- Main Chat Interface ---
# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input at the bottom of the screen
if prompt := st.chat_input("Ask a question about your document (e.g., 'What is the summary of page 2?')..."):
    
    # Display user message and add to session state
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate AI response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing document..."):
            response = ru.get_answer(prompt)
            st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})