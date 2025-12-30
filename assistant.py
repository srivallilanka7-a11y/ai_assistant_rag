import ollama
import os

MEMORY_FILE = "memory.txt"

def load_memory():
    memory = {}
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r") as f:
            for line in f:
                if "=" in line:
                    key, value = line.strip().split("=", 1)
                    memory[key] = value
    return memory

def save_memory(key, value):
    memory = load_memory()
    memory[key] = value
    with open(MEMORY_FILE, "w") as f:
        for k, v in memory.items():
            f.write(f"{k}={v}\n")

def ask_ai(question):
    q = question.lower().strip()
    memory = load_memory()

    # SAVE NAME
    if q.startswith("remember my name is"):
        name = q.replace("remember my name is", "").strip()
        save_memory("name", name)
        return f"✅ I will remember your name is {name}."

    # ANSWER NAME
    if q in ["what is my name", "tell me my name"]:
        if "name" in memory:
            return f"Your name is {memory['name']}."
        else:
            return "I don't know your name yet. Please tell me."

    # NORMAL AI RESPONSE
    context = "\n".join([f"{k}: {v}" for k, v in memory.items()])

    prompt = f"""
You are a personal offline AI assistant.

Known user information:
{context}

User question:
{question}
"""

    response = ollama.chat(
        model="phi3",
        messages=[{"role": "user", "content": prompt}],
        options={"num_predict": 150}
    )

    return response["message"]["content"]
