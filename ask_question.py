from pymongo import MongoClient
from sentence_transformers import SentenceTransformer

# MongoDB connection
client = MongoClient(
    "mongodb+srv://srivallilanka7_db_user:valli123@cluster0.whh3qij.mongodb.net/?appName=Cluster0"
)

db = client["rag_db"]
collection = db["documents"]

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Ask user question
question = input("Ask your question: ")

# Convert question to embedding
query_embedding = model.encode(question).tolist()

# Vector search
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

print("\n🔍 Answer from database:\n")
for doc in results:
    print("-", doc["text"])
