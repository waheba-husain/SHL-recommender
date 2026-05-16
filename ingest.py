import json, os, pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"   # fast, free, good quality

def build_index():
    with open("catalog.json") as f:
        catalog = json.load(f)
    
    model = SentenceTransformer(MODEL_NAME)
    
    # Build rich text for each product (name + description + test types)
    texts = []
    for item in catalog:
        types_str = ", ".join(item.get("test_types", [])) or "assessment"
        text = f"{item['name']}. {item.get('description', '')} Test type: {types_str}"
        texts.append(text)
    
    print(f"Embedding {len(texts)} products...")
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)   # Inner product = cosine similarity (normalized)
    index.add(embeddings.astype("float32"))
    
    os.makedirs("faiss_index", exist_ok=True)
    faiss.write_index(index, "faiss_index/index.faiss")
    with open("faiss_index/metadata.json", "w") as f:
        json.dump(catalog, f, indent=2)
    
    print(f"Index built: {index.ntotal} vectors, dim={dim}")

if __name__ == "__main__":
    build_index()