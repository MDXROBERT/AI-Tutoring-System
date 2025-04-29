from sentence_transformers import SentenceTransformer
import faiss
import json
import numpy as np
import os


embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def create_embeddings():
    
    with open("data/training_data.json", "r", encoding="utf-8") as f:
        content_data = json.load(f)
    
    
    question_data = []
    question_file = "data/training_questions.json"
    if os.path.exists(question_file):
        with open(question_file, "r", encoding="utf-8") as f:
            question_data = json.load(f)
    
    
    content_texts = [f"{item['topic']} - {item['subtopic']}: {item['summary']} {item['detailed_content']}" 
                   for item in content_data]
    
    
    question_texts = [f"Question about {item['topic']}: {item['question']}" 
                    for item in question_data]
    
    
    all_texts = content_texts + question_texts
    
    print(f"Creating embeddings for {len(content_texts)} content items and {len(question_texts)} questions")
    
    
    embeddings = embedding_model.encode(all_texts, convert_to_numpy=True)

   
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    
    faiss.write_index(index, "data/course_embeddings.index")
    
    
    mapping = {
        "content_count": len(content_texts),
        "question_count": len(question_texts),
        "total_count": len(all_texts)
    }
    with open("data/embeddings_mapping.json", "w", encoding="utf-8") as f:
        json.dump(mapping, f)
    
    print(" FAISS Embeddings Created & Saved!")
    print(f" Embedding mapping saved to data/embeddings_mapping.json")

if __name__ == "__main__":
    create_embeddings()