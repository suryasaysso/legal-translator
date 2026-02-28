# Main Streamlit App for Legal Translator (Completely vibe coded)

import streamlit as st
import os
from src.ingest import process_pdf_to_chroma
from src.model_loader import get_deepseek_llm, clean_reasoning
from src.translator import get_tos_chain

# --- Page Config ---
st.set_page_config(page_title="Legal Translator", page_icon="⚖️", layout="wide")
st.title("⚖️ Legal Translator")
st.markdown("Upload a Terms of Service PDF and let **DeepSeek-R1** find the hidden risks.")

# --- Sidebar: Model Status ---
with st.sidebar:
    st.header("Settings")
    status = st.empty()
    if st.button("Clear Vector Database"):
        if os.path.exists("./chroma_db"):
            import shutil
            shutil.rmtree("./chroma_db")
            st.success("Database cleared!")
        else:
            st.info("No database found.")

# --- 1. File Upload ---
uploaded_file = st.file_uploader("Upload Terms of Service PDF", type="pdf")

if uploaded_file:
    # Save file temporarily to disk for processing
    temp_path = os.path.join("data", uploaded_file.name)
    os.makedirs("data", exist_ok=True)
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # --- 2. Ingestion ---
    with st.spinner("Analyzing PDF and creating vector embeddings..."):
        vector_db = process_pdf_to_chroma(temp_path)
        st.success(f"Successfully ingested: {uploaded_file.name}")

    # --- 3. Chat Interface ---
    st.divider()
    user_input = st.text_input("Ask a question (e.g., 'What are the refund rules?' or 'How is my data shared?')")

    if user_input:
        with st.spinner("DeepSeek is thinking..."):
            # Initialize Brain & Chain
            llm = get_deepseek_llm()
            chain = get_tos_chain(llm, vector_db)
            
            # Execute RAG
            response = chain.invoke({"input": user_input})
            
            # Clean DeepSeek's <think> tags for the UI
            final_answer = clean_reasoning(response["answer"])
            
            # Display Result
            st.subheader("Analysis")
            st.markdown(final_answer)
            
            # Optional: Show Sources
            with st.expander("View Source Snippets"):
                for doc in response["context"]:
                    st.write(f"--- \n {doc.page_content}")