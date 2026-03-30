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
    reader = PdfReader(upload_file)

    st.subheader("Dataset Preview")
    st.write("File uploaded successfully")


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
    st.subheader("Ask a question:")
    query = st.text_input()

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
