import streamlit as st
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from transformers import pipeline

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="My RAG AI Assistant", layout="centered")
st.title("🤖 My RAG AI Assistant (Low-RAM Safe)")

# ---------------- MONGODB CONNECTION ----------------
client = MongoClient(
    "mongodb+srv://srivallilanka7_db_user:valli123@cluster0.whh3qij.mongodb.net/?appName=Cluster0"
)

db = client["rag_db"]
collection = db["documents"]

# ---------------- LOAD MODELS ----------------
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def load_generator():
    return pipeline(
        "text2text-generation",
        model="google/flan-t5-small"
    )

embed_model = load_embedder()
generator = load_generator()

# ---------------- CHUNK FUNCTION ----------------
def chunk_text(text, chunk_size=300):
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

# ---------------- PDF UPLOAD ----------------
st.subheader("📄Upload PDF")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file is not None:
    reader = PdfReader(uploaded_file)
    pdf_text = ""

    for page in reader.pages:
        text = page.extract_text()
        if text:
            pdf_text += text + " "

    st.success("PDF text extracted successfully")

    if st.button("Store PDF in Database"):
        chunks = chunk_text(pdf_text)

        for chunk in chunks:
            embedding = embed_model.encode(chunk).tolist()
            collection.insert_one({
                "text": chunk,
                "embedding": embedding
            })

        st.success(f"PDF stored successfully in {len(chunks)} chunks")

st.divider()

# ---------------- QUESTION ANSWERING ----------------
st.subheader("❓ Ask Questions")

question = st.text_input("Ask your question:")

if st.button("Ask"):
    if question.strip() == "":
        st.warning("Please enter a question")
    else:
        # Convert question to embedding
        query_embedding = embed_model.encode(question).tolist()

        # Vector search in MongoDB
        results = collection.aggregate([
            {
                "$vectorSearch": {
                    "index": "vector_index",
                    "path": "embedding",
                    "queryVector": query_embedding,
                    "numCandidates": 100,
                    "limit": 3
                }
            }
        ])

        results_list = list(results)
        context = ""

        if len(results_list) == 0:
            st.warning("No relevant answer found")
        else:
            # Combine retrieved text
            context = " ".join([doc["text"] for doc in results_list])

            # ---------- CLEAN ANSWER PROMPT ----------
                # --------- CLEAN ANSWER PROMPT ---------
        prompt = f"""
Question: {question}

Context:
{context}

Instructions:
1. If the context is useful, use it.
2. If it is not relevant, ignore it and answer from your knowledge.
3. Give a short, simple, correct answer.
4. Do NOT repeat the question.
5. Do NOT say you are an expert teacher.

Answer:
"""
        response = generator(prompt, max_new_tokens=200, do_sample=False)
        final_answer = response[0]["generated_text"]

        st.subheader("Best Answer")
        st.write(final_answer)



         
