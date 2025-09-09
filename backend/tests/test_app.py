
import streamlit as st
import asyncio
import os
from openai import OpenAI
from pathlib import Path
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from services.document_processor import DocumentProcessor
from services.rag_pipeline import RAGPipeline
from dotenv import load_dotenv
import os

load_dotenv()

# ------------------------------
# Initialize processors
# ------------------------------
@st.cache_resource
def init_processors():
    # Make sure to set OPENAI_API_KEY in environment variables
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("OPENAI_API_KEY is not set in environment variables!")
        st.stop()
    
    client = OpenAI(api_key=api_key)
    doc_processor = DocumentProcessor()
    rag_pipeline = RAGPipeline(openai_client=client)
    return doc_processor, rag_pipeline

doc_processor, rag_pipeline = init_processors()

# ------------------------------
# Streamlit UI
# ------------------------------
st.title("RAG System Testing Interface")

# File uploader
uploaded_file = st.file_uploader("Upload a document", type=["txt", "pdf", "xlsx", "xls"])

if uploaded_file:
    st.info(f"Processing {uploaded_file.name} ...")
    
    async def process_file_and_query(file):
        # Process the uploaded file
        doc_result = await doc_processor.process_file(file)
        
        # Ask a question to RAG pipeline
        question = st.text_input("Enter a question about the document:")
        answer_result = None
        if question:
            answer_result = await rag_pipeline.process_query(question)
        return doc_result, answer_result

    # Run the async processing
    doc_result, answer_result = asyncio.run(process_file_and_query(uploaded_file))

    # Show extracted entities
    st.subheader("Document Processing Result")
    st.write(f"Filename: {doc_result['filename']}")
    st.write("Extracted Entities:")
    for entity_type, entities in doc_result["entities"].items():
        st.write(f"- {entity_type}: {', '.join(entities)}")

    # Show RAG answer
    if answer_result:
        st.subheader("RAG Pipeline Result")
        st.write(f"Answer: {answer_result['answer']}")
        st.write(f"Confidence: {answer_result['confidence']}")
        if answer_result["sources"]:
            st.write("Sources:")
            for src in answer_result["sources"]:
                st.write(f"- {src}")
