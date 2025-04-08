"""A simple PDF Querying system using RAG(Retrieval-Augmented Generation)
Created by: Shadan Alam
Date: 8th April 2025.
"""
#----------------------------------------------------------------< Imporing Necessary Libraries >
import streamlit as st
import os
import re
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from tqdm.auto import tqdm
#---------------------------------------------------------------< Custom Semantic Splitting class >
class QASemanticSplitter:
    def __init__(self):
        pass
    def split_text(self, text: str) -> List[str]:
        # Basic cleanups
        text = re.sub(r'\n{3,}', '\n\n', text)  # Collapse multiple newlines
        text = re.sub(r'===== Page \d+ =====', '', text)  # Remove manual page markers

        # Remove footer lines like: www.cpp-programs.blogspot.com Page 10
        text = re.sub(r'www\.[\w\.-]+\.com\s+Page\s+\d+', '', text, flags=re.IGNORECASE)

        # Remove standalone "Page X" patterns
        text = re.sub(r'\bPage\s+\d+\b', '', text, flags=re.IGNORECASE)

        # Pattern to match question starts
        question_pattern = re.compile(
            r'(?=\n*(\d+\s*[\.\)]|Q:|Question:|QUESTION:))', re.IGNORECASE
        )

        # Split based on detected question patterns
        chunks = question_pattern.split(text)

        # Grouping chunks into question-answer pairs
        qa_pairs = []
        current_question = ""

        for i in range(len(chunks)):
            part = chunks[i].strip()
            if not part:
                continue

            # If this part looks like a new question, store the previous QA if it exists
            if re.match(r'^(\d+\s*[\.\)]|Q:|Question:|QUESTION:)', part, re.IGNORECASE):
                if current_question:
                    qa_pairs.append(current_question.strip())
                current_question = part
            else:
                # It's part of the current answer — keep appending
                current_question += "\n" + part

        # Add the last question-answer if any
        if current_question:
            qa_pairs.append(current_question.strip())

        return qa_pairs
#-----------------------------------------------------------------< Initializing Pinecone Vector Database >
def init_pinecone():
    pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
    index_name = "cprogramming"
    dimension = 384  # bge-small-en-v1.5 embedding dimension
    
    # Check if index exists, create if not
    existing_indexes = pc.list_indexes().names()
    if index_name not in existing_indexes:
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
    
    return pc.Index(index_name)
#------------------------------------------------------------------------------<PDF Processing>
def process_pdfs(uploaded_files, index):
    model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    splitter = QASemanticSplitter()
    
    # Create temp directory if not exists
    if not os.path.exists("temp_pdfs"):
        os.makedirs("temp_pdfs")
    
    all_chunks = []
    
    for uploaded_file in uploaded_files:
        # Save uploaded file to temp directory
        file_path = os.path.join("temp_pdfs", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Load and split PDF
        loader = PyPDFLoader(file_path)
        pages = loader.load()
        
        for doc in pages:
            # Clean content
            clean_content = doc.page_content.replace('\r', '')
            clean_content = re.sub(r'===== Page \d+ =====', '', clean_content)
            clean_content = re.sub(r'\n{3,}', '\n\n', clean_content).strip()
            
            # Split into Q&A chunks
            chunks = splitter.split_text(clean_content)
            
            # Create document objects
            for chunk in chunks:
                if len(chunk) > 10:  # Skip very small chunks
                    all_chunks.append({
                        "page_content": chunk,
                        "metadata": {
                            "source": uploaded_file.name,
                            "page": doc.metadata["page"]
                        }
                    })
    
    # Index chunks in batches
    batch_size = 10
    for i in tqdm(range(0, len(all_chunks), batch_size)):
        batch = all_chunks[i:i + batch_size]
        texts = [chunk["page_content"] for chunk in batch]
        embeddings = model.encode(texts, show_progress_bar=False)
        
        metadata_batch = []
        for chunk in batch:
            parts = chunk["page_content"].split("\n", 1)
            question = parts[0].strip()
            answer = parts[1].strip() if len(parts) > 1 else ""
            
            metadata_batch.append({
                "text": chunk["page_content"],
                "source": chunk["metadata"]["source"],
                "page": chunk["metadata"]["page"],
                "question": question,
                "answer": answer
            })
        
        ids = [f"chunk_{i + j}" for j in range(len(batch))]
        vectors = list(zip(ids, embeddings.tolist(), metadata_batch))
        index.upsert(vectors=vectors)
    
    return len(all_chunks)
#---------------------------------------------------------------< Query your prompt >
def query_qna(question, index, top_k=1, similarity_threshold=0.7):
    model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    query_embed = model.encode(question).tolist()
    
    results = index.query(
        vector=query_embed,
        top_k=top_k,
        include_metadata=True
    )
    
    # Filter results by similarity score threshold
    filtered_results = [
        {
            "score": match.score,
            "question": match.metadata["question"],
            "answer": match.metadata["answer"],
            "source": match.metadata["source"],
            "page": int(match.metadata["page"]) + 1
        } 
        for match in results.matches 
        if match.score >= similarity_threshold
    ]
    
    # Return "Sorry!" message if no relevant results found
    if not filtered_results:
        return [{
            "question": question,
            "answer": "❌Sorry! This PDF doesn't contain the answer to your question.",
            "score": 0,
            "source": "N/A",
            "page": 0
        }]
    
    return filtered_results
#---------------------------------------------------------------< Interface>
def main():
    st.set_page_config(page_title="PDF Query System", layout="wide")
    
    # Initialize Pinecone
    index = init_pinecone()
    
    st.title("📑📚Multi PDFs Querying system using RAG🧠")
    st.markdown("Upload a single or multiple PDF documents and ask questions from the content")
    
    # File upload section
    with st.expander("📤 Upload PDF Documents", expanded=True):
        uploaded_files = st.file_uploader(
            "Choose PDF files.", 
            type="pdf", 
            accept_multiple_files=True,
            help="Upload one or more PDF documents to build the knowledge base"
        )
        
        if uploaded_files and st.button("⚙️Process PDFs"):
            with st.spinner("Processing files please wait"):
                num_chunks = process_pdfs(uploaded_files, index)
                st.success(f"🟢{len(uploaded_files)} PDF(s) processed successfully!")
    
    # Query section
    st.divider()
    st.subheader("🔍Ask Questions About Your Documents")
    
    question = st.text_input(
        "Enter your question:",
        placeholder="Enter your question here.",
        help="Ask any question about the content of your uploaded PDFs"
    )
    
    if question and st.button("Search🔍"):
        if not uploaded_files:
            st.warning("⚠️Please upload PDF documents first")
        else:
            with st.spinner("Searching for answers please wait...⏳"):
                # In your Streamlit app where you display results:
                results = query_qna(question, index)

                for i, result in enumerate(results, 1):
                    st.markdown(f"### Question {result['question']}")
    
                    if result['score'] == 0:  # This is our "Sorry!" response
                        st.warning(result['answer'])
                    else:
                        st.success(f"Confidence: {result['score']:.1%}")
                        st.markdown(f"{result['answer']}")
                        st.markdown(f"**Source:** {result['source']}  \n**Page:** {result['page']}", unsafe_allow_html=True)

    # Clear cache button
    st.divider()
    col1, col2 = st.columns([1,5])
    with col1:
        if st.button("Clear inputs🔄"):
        # Clear temp files
            if os.path.exists("temp_pdfs"):
                for file in os.listdir("temp_pdfs"):
                    os.remove(os.path.join("temp_pdfs", file))
                    os.rmdir("temp_pdfs")
                    st.cache_data.clear()
                    st.rerun()
    with col2:
        if st.button("Exit❌"):
            st.write("Exiting app. Thank You!!😊")
            st.stop() 
main()