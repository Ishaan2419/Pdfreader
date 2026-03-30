
"""import streamlit as st
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from transformers import pipeline
import warnings

warnings.filterwarnings("ignore")

st.title("📄 PDF Chatbot (RAG)")
st.write("Upload PDF → Ask Questions → Get Answers")

uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

@st.cache_resource
def load_models():
    embed_model = SentenceTransformer("all-mpnet-base-v2")
    generator = pipeline("text-generation", model="gpt2")
    return embed_model, generator

embed_model, generator = load_models()

if uploaded_file is not None:
    reader = PdfReader(uploaded_file)

    text = ""
    for page in reader.pages:
        text += page.extract_text()

    chunks = text.split("\n")

   
    embeddings = embed_model.encode(chunks)
    embeddings = np.array(embeddings).astype("float32")

 
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    st.success("✅ PDF processed successfully!")

    query = st.text_input("Ask a question:")

    if query:
        # Encode query
        query_embedding = embed_model.encode([query]).astype("float32")

        # Search
        k = 2
        distances, indices = index.search(query_embedding, k)

        retrieved = [chunks[i] for i in indices[0]]
        context = " ".join(retrieved)

        # Prompt
        prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:\n"

        # Generate answer
        result = generator(
            prompt,
            max_new_tokens=50,
            do_sample=False,
            pad_token_id=50256
        )

        output = result[0]['generated_text']

        answer = output[len(prompt):]
        answer = answer.split("Question:")[0]
        answer = answer.split("Q:")[0]

        # Display
        st.subheader("🧠 Answer:")
        st.write(answer.strip())
        """

import streamlit as st
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from transformers import pipeline
import warnings

warnings.filterwarnings("ignore")

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="PDF Chatbot", page_icon="📄", layout="wide")

# -------------------------------
# CUSTOM CSS (🔥 MAIN UPGRADE)
# -------------------------------
st.markdown("""
    <style>
    .main {
        background-color: #0E1117;
        color: white;
    }
    .title {
        text-align: center;
        font-size: 40px;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #9CA3AF;
        margin-bottom: 20px;
    }
    .stTextInput>div>div>input {
        background-color: #1F2937;
        color: white;
        border-radius: 10px;
        padding: 10px;
    }
    .answer-box {
        background-color: #1F2937;
        padding: 15px;
        border-radius: 10px;
        margin-top: 10px;
    }

   
    </style>
""", unsafe_allow_html=True)

# -------------------------------
# HEADER
# -------------------------------
st.markdown('<div class="title">📄 PDF Chatbot (RAG)</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload PDF → Ask Questions → Get Smart Answers</div>', unsafe_allow_html=True)

# -------------------------------
# LOAD MODELS (CACHED)
# -------------------------------
@st.cache_resource
def load_models():
    embed_model = SentenceTransformer("all-mpnet-base-v2")
    generator = pipeline("text-generation", model="gpt2")
    return embed_model, generator

embed_model, generator = load_models()

# -------------------------------
# LAYOUT (2 COLUMNS 🔥)
# -------------------------------
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📂 Upload PDF")
    uploaded_file = st.file_uploader("", type="pdf")

with col2:
    st.subheader("💬 Ask Questions")

    if uploaded_file is not None:
        reader = PdfReader(uploaded_file)

        text = ""
        for page in reader.pages:
            text += page.extract_text()

        chunks = text.split("\n")

        embeddings = embed_model.encode(chunks)
        embeddings = np.array(embeddings).astype("float32")

        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)

        st.success("✅ PDF processed successfully!")

        # Chat input
        query = st.text_input("Type your question...")

        if query:
            query_embedding = embed_model.encode([query]).astype("float32")

            k = 2
            distances, indices = index.search(query_embedding, k)

            retrieved = [chunks[i] for i in indices[0]]
            context = " ".join(retrieved)

            prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:\n"

            result = generator(
                prompt,
                max_new_tokens=60,
                do_sample=False,
                pad_token_id=50256
            )

            output = result[0]['generated_text']

            answer = output[len(prompt):]
            answer = answer.split("Question:")[0]
            answer = answer.split("Q:")[0]

            # -------------------------------
            # DISPLAY LIKE CHAT 💬
            # -------------------------------
            st.markdown("### 🧑 You:")
            st.markdown(f"**{query}**")

            st.markdown("### 🤖 AI:")
            st.markdown(f'<div class="answer-box">{answer.strip()}</div>', unsafe_allow_html=True)

    else:
        st.info("👈 Upload a PDF to start chatting")
