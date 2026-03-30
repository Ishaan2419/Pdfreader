import streamlit as st
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from transformers import pipeline
import warnings

warnings.filterwarnings("ignore")

@st.cache_resource
def load_models():
    embed_model = SentenceTransformer("all-mpnet-base-v2")
    generator = pipeline("text-generation", model="gpt2")
    return embed_model, generator

embed_model, generator = load_models()
st.header("PDF Chatbot (RAG)")
st.write("Upload the PDF to Ask Questions")

st.sidebar.header("Upload Dataset")
upload_file = st.sidebar.file_uploader("Upload The PDF File", type="pdf")

if upload_file is not None:
    df = pdfReader(upload_file)

    st.subheader("Dataset Preview")
    st.write("File uploaded successfully")
