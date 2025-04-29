from sentence_transformers import SentenceTransformer
import faiss
import json

def load_model():
    
    model = SentenceTransformer("all-MiniLM-L6-v2")
    index = faiss.read_index("data/course_embeddings.index")
    
    with open("data/training_data.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    
    return model, index, data

def search_database(query):
   
    model, index, data = load_model()
    query_embedding = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, k=3)
    
    results = [data[i] for i in indices[0]]
    return results
