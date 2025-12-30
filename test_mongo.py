from pymongo import MongoClient

client = MongoClient(
    "mongodb+srv://srivallilanka7_db_user:valli123@cluster0.whh3qij.mongodb.net/?appName=Cluster0"
)

print(client.list_database_names())
