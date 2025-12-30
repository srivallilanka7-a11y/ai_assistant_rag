from pymongo import MongoClient
from sentence_transformers import SentenceTransformer

# 1️⃣ MongoDB connection
mongo_client = MongoClient(
    "mongodb+srv://srivallilanka7_db_user:valli123@cluster0.whh3qij.mongodb.net/?appName=Cluster0")

db = mongo_client["rag_db"]
collection = db["documents"]

# 2️⃣ Load FREE local embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# 3️⃣ Text to store
text = "MongoDB Atlas is a cloud database used for scalable applications."

# 4️⃣ Convert text → embedding (FREE)
embedding = model.encode(text).tolist()

# 5️⃣ Store in MongoDB
collection.insert_one({
    "text": text,
    "embedding": embedding
})

print("✅ Data inserted using LOCAL embeddings")
