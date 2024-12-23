import streamlit as st
import os
from dotenv import load_dotenv
from utils import (
    extract_text_from_pdf,
    build_hierarchical_tree,
    save_tree,
    hybrid_retrieval,
    rag_answer,
)

# Load API key from .env
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Create necessary directories
os.makedirs("uploaded_textbooks", exist_ok=True)
os.makedirs("hierarchical_trees", exist_ok=True)
os.makedirs("retrieved_contexts", exist_ok=True)

# Streamlit UI
st.title("Hierarchical Question-Answering System 📚🤖")
st.markdown(
    "Upload textbooks, explore their structure, and ask questions powered by AI."
)

# Upload PDF section
uploaded_files = st.file_uploader("Upload Textbooks (PDF)", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        file_path = os.path.join("uploaded_textbooks", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())

        # Extract text
        st.write(f"Processing: {uploaded_file.name}")
        extracted_text = extract_text_from_pdf(file_path)

        # Build hierarchical tree
        tree = build_hierarchical_tree(extracted_text, textbook_title=uploaded_file.name)
        tree_path = os.path.join("hierarchical_trees", f"{uploaded_file.name}_tree.json")
        save_tree(tree, tree_path)

        st.success(f"Processed and indexed: {uploaded_file.name}")

# Query Section
query = st.text_input("Ask a question:")
if query:
    st.write("Retrieving relevant information...")
    relevant_text = hybrid_retrieval(query, OPENAI_API_KEY)
    if relevant_text:
        st.write("Generating an answer...")
        answer = rag_answer(query, relevant_text, OPENAI_API_KEY)
        st.write(f"**Answer:** {answer}")
        st.write("**Relevant Context:**")
        st.write(relevant_text)
    else:
        st.write("No relevant information found.")
