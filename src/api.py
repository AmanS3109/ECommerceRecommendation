from fastapi import FastAPI, HTTPException
import numpy as np
import time
import json
from collections import defaultdict

app = FastAPI(title="E-Commerce Recommendation Engine")

# Global variables for RAM
P = None
Q = None
item_metadata = {}
user_history = defaultdict(list) # Stores what each user already bought

@app.on_event("startup")
async def load_matrices():
    global P, Q, item_metadata, user_history
    print("Loading ML matrices, Metadata, and History into RAM...")
    try:
        P = np.load("data/trained_P.npy")
        Q = np.load("data/trained_Q.npy")
        
        with open("data/item_metadata.json", "r") as f:
            raw_metadata = json.load(f)
            item_metadata = {int(k): v for k, v in raw_metadata.items()}
            
        # NEW: Load the historical interactions so we know what to filter
        interactions = np.load("data/jax_interactions.npy")
        for user_idx, item_idx in interactions:
            user_history[int(user_idx)].append(int(item_idx))
            
        print("System Online! Ready to serve traffic.")
    except Exception as e:
        print(f"Startup Error: {e}")

@app.get("/recommend/{user_idx}")
async def get_recommendations(user_idx: int, top_k: int = 10):
    if user_idx < 0 or user_idx >= P.shape[0]:
        raise HTTPException(status_code=404, detail="User index out of bounds")

    start_time = time.time()
    
    user_vector = P[user_idx]
    all_scores = np.dot(user_vector, Q.T)
    
    # NEW: The Seen Items Filter
    # Get the items the user already bought (default to empty list if none)
    seen_items = user_history.get(user_idx, [])
    
    # Force the score of already-purchased items to negative infinity 
    # so they mathematically cannot appear in the top K
    if seen_items:
        all_scores[seen_items] = -np.inf
    
    # Sort and slice as normal
    top_indices = np.argsort(all_scores)[-top_k:][::-1]
    top_scores = all_scores[top_indices]
    
    # Hydration
    recommendations = []
    for idx, score in zip(top_indices, top_scores):
        meta = item_metadata.get(int(idx), {"original_id": "Unknown", "product_name": "Unknown"})
        recommendations.append({
            "internal_idx": int(idx),
            "product_id": meta["original_id"],
            "category": meta["product_name"],
            "match_score": float(score)
        })
        
    end_time = time.time()
    
    return {
        "user_idx": user_idx,
        "compute_time_ms": round((end_time - start_time) * 1000, 2),
        "filtered_items_count": len(seen_items), # Show recruiters the filter is working!
        "recommendations": recommendations
    }