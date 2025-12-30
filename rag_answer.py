from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# 1️⃣ MongoDB connection
client = MongoClient(
    "mongodb+srv://srivallilanka7_db_user:valli123@cluster0.whh3qij.mongodb.net/?appName=Cluster0"
)

db = client["rag_db"]
collection = db["documents"]

# 2️⃣ Load embedding model (same as before)
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# 3️⃣ Load FREE text generation model
generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-small"
)

# 4️⃣ Ask question
question = input("Ask your question: ")

# 5️⃣ Convert question → embedding
query_embedding = embed_model.encode(question).tolist()

# 6️⃣ Retrieve relevant text from MongoDB
results = collection.aggregate([
    {
        "$vectorSearch": {
            "index": "vector_index",
            "path": "embedding",
            "queryVector": query_embedding,
            "numCandidates": 100,
            "limit": 2
        }
    }
])

context = ""
for doc in results:
    context += doc["text"] + " "

# 7️⃣ Combine question + context
prompt = f"""
Answer the question using the context below.

Context:
{context}

Question:
{question}

Answer:
"""

# 8️⃣ Generate answer
response = generator(prompt, max_length=150)

print("\n🤖 AI Answer:\n")
print(response[0]["generated_text"])
