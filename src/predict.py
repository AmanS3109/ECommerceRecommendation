import numpy as np
import time

def recommend_for_user(user_idx=42, top_k=10):
    print(f"Loading trained matrices from disk...")
    # Load the optimized matrices we just saved
    P = np.load("data/trained_P.npy")
    Q = np.load("data/trained_Q.npy")
    
    print(f"Generating recommendations for User {user_idx}...")
    start_time = time.time()
    
    # 1. Get the specific user's 64-dimensional preference vector
    user_vector = P[user_idx]
    
    # 2. THE DOT PRODUCT (The core engine)
    # We multiply this single user vector against ALL 32,951 item vectors at once.
    # NumPy handles this via highly optimized C-level linear algebra.
    all_item_scores = np.dot(user_vector, Q.T)
    
    # 3. Sort the scores to find the highest matches
    # argsort sorts lowest-to-highest, so we take the last 'top_k' and reverse it
    top_item_indices = np.argsort(all_item_scores)[-top_k:][::-1]
    top_item_scores = all_item_scores[top_item_indices]
    
    end_time = time.time()
    
    print(f"\n--- Top {top_k} Recommendations ---")
    for rank, (item_idx, score) in enumerate(zip(top_item_indices, top_item_scores)):
        print(f"Rank {rank+1}: Item ID {item_idx:5d} | Match Score: {score:.4f}")
        
    print(f"\nCalculated in {(end_time - start_time) * 1000:.2f} milliseconds.")

if __name__ == "__main__":
    # Feel free to change this number to any user index between 0 and 95419
    recommend_for_user(user_idx=100)