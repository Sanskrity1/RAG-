import ollama
from .config import OLLAMA_LLM_MODEL

def answer_question(context, query):
    prompt = f"Answer the question based on the following movie data:\n{context}\n\nQuestion: {query}\nAnswer:"
    
    try:
        response = ollama.chat(
            model=OLLAMA_LLM_MODEL,
            messages=[{"role": "user", "content": prompt}]
        )
        return response["content"]
    except Exception as e:
        print(f"Error generating answer: {e}")
        return "Could not generate an answer."
