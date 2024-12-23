import PyPDF2
import json
import networkx as nx
from sentence_transformers import SentenceTransformer, util
import openai

# Model for embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")

# 1. Extract Text from PDF
def extract_text_from_pdf(file_path):
    """Extract text from a PDF."""
    text = ""
    with open(file_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text()
    return text

# 2. Build Hierarchical Tree
def build_hierarchical_tree(text, textbook_title):
    """Create a hierarchical tree structure."""
    lines = text.split("\n")
    tree = {"title": textbook_title, "chapters": []}
    current_chapter = None

    for line in lines:
        if line.strip().startswith("Chapter"):
            current_chapter = {"title": line.strip(), "sections": []}
            tree["chapters"].append(current_chapter)
        elif current_chapter and line.strip():
            current_chapter["sections"].append(line.strip())
    return tree

def save_tree(tree, path):
    """Save the hierarchical tree."""
    with open(path, "w") as f:
        json.dump(tree, f, indent=4)

# 3. Hybrid Retrieval
def hybrid_retrieval(query, openai_api_key):
    """Retrieve relevant text using hybrid methods."""
    with open("hierarchical_trees/example_tree.json") as f:  # Adjust file path as needed
        tree = json.load(f)
    
    all_sections = [
        section for chapter in tree["chapters"] for section in chapter["sections"]
    ]
    query_embedding = model.encode(query, convert_to_tensor=True)
    section_embeddings = model.encode(all_sections, convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(query_embedding, section_embeddings)
    
    top_indices = similarities[0].topk(3).indices.tolist()
    return " ".join([all_sections[i] for i in top_indices])

# 4. RAG Answer Generation
def rag_answer(query, context, openai_api_key):
    """Generate an answer using Retrieval-Augmented Generation."""
    openai.api_key = openai_api_key
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=f"Answer the question based on the context below:\n\nContext: {context}\n\nQuestion: {query}\n\nAnswer:",
        max_tokens=150,
    )
    return response.choices[0].text.strip()
