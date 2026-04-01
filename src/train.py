import jax
import jax.numpy as jnp
from jax import random
import numpy as np
import optax
from jax import value_and_grad

# 1. System Parameters
NUM_USERS = 95420
NUM_ITEMS = 32951
LATENT_DIM = 64  # 'k' hidden features

def init_network(seed=42):
    """
    Allocates memory for our User (P) and Item (Q) matrices
    filled with small random numbers.
    """
    print("Initializing JAX Embeddings...")
    key = random.PRNGKey(seed)
    key_p, key_q = random.split(key)

    # We scale down the random numbers by 0.1 so the initial dot products 
    # aren't massively huge, which would cause the gradients to explode.
    P = random.normal(key_p, (NUM_USERS, LATENT_DIM)) * 0.1
    Q = random.normal(key_q, (NUM_ITEMS, LATENT_DIM)) * 0.1
    
    return P, Q

@jax.jit
def bpr_loss(P, Q, user_indices, pos_item_indices, neg_item_indices, l2_reg=0.0001):
    """
    The core math engine. 
    Calculates how wrong the current matrices are based on a batch of data.
    """
    # 1. Memory Lookup: Fetch the specific rows for this batch
    user_vecs = P[user_indices]           # shape: [batch_size, 64]
    pos_item_vecs = Q[pos_item_indices]   # shape: [batch_size, 64]
    neg_item_vecs = Q[neg_item_indices]   # shape: [batch_size, 64]

    # 2. The Forward Pass (Dot Products)
    # We multiply the user vectors by the item vectors element-wise, 
    # then sum across the 64 features to get a single score per user-item pair.
    pos_scores = jnp.sum(user_vecs * pos_item_vecs, axis=1)
    neg_scores = jnp.sum(user_vecs * neg_item_vecs, axis=1)

    # 3. The Difference
    # We want the positive score to be MUCH higher than the negative score.
    differences = pos_scores - neg_scores

    # 4. The Loss Function (Log Sigmoid)
    # jax.nn.log_sigmoid penalizes the model heavily if 'difference' is negative.
    # We take the mean across the whole batch, and make it negative because 
    # optimizers always try to MINIMIZE a number.
    loss = -jnp.mean(jax.nn.log_sigmoid(differences))

    # 5. Regularization (Penalty for huge numbers)
    # This prevents the vectors from growing to infinity.
    reg_loss = l2_reg * (jnp.sum(user_vecs**2) + jnp.sum(pos_item_vecs**2) + jnp.sum(neg_item_vecs**2))

    return loss + reg_loss


def generate_batches(interactions, num_items, batch_size=1024):
    """
    Takes the real interactions and dynamically generates random negative samples.
    Yields batches of (users, pos_items, neg_items).
    """
    # 1. Shuffle the data. SGD requires random ordering to converge properly.
    np.random.shuffle(interactions)
    
    num_batches = len(interactions) // batch_size
    
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        
        batch = interactions[start_idx:end_idx]
        
        user_indices = batch[:, 0]
        pos_item_indices = batch[:, 1]
        
        # 2. THE MAGIC: Negative Sampling
        # We generate random integers between 0 and NUM_ITEMS.
        # This executes in microseconds and takes almost zero memory.
        neg_item_indices = np.random.randint(0, num_items, size=batch_size)
        
        yield user_indices, pos_item_indices, neg_item_indices

def train_model():
    print("Loading data...")
    interactions = np.load("data/jax_interactions.npy")
    
    # Initialize our random Embeddings
    P, Q = init_network()
    
    # Setup the Adam optimizer (a smarter version of basic Gradient Descent)
    learning_rate = 0.001
    optimizer = optax.adam(learning_rate)
    
    # We combine P and Q into a single dictionary so the optimizer can update both
    params = {'P': P, 'Q': Q}
    opt_state = optimizer.init(params)
    
    # JAX Magic: value_and_grad returns both the Loss AND the Gradients
    # We tell it to calculate gradients with respect to 'params' (argnum 0)
    loss_and_grad_fn = jax.jit(value_and_grad(
        lambda p, u, pos, neg: bpr_loss(p['P'], p['Q'], u, pos, neg), 
        argnums=0
    ))
    
    EPOCHS = 30
    BATCH_SIZE = 2048
    
    print("Starting Training Loop...")
    for epoch in range(EPOCHS):
        total_loss = 0.0
        batches = 0
        
        # 1. Generate batches with dynamically sampled negatives
        batch_generator = generate_batches(interactions, NUM_ITEMS, BATCH_SIZE)
        
        for users, pos_items, neg_items in batch_generator:
            # 2. Convert numpy arrays to JAX arrays
            u_jax = jnp.array(users)
            pos_jax = jnp.array(pos_items)
            neg_jax = jnp.array(neg_items)
            
            # 3. Calculate Loss and Gradients
            loss_val, grads = loss_and_grad_fn(params, u_jax, pos_jax, neg_jax)
            
            # 4. Update the matrices using the optimizer
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            
            total_loss += loss_val
            batches += 1
            
        avg_loss = total_loss / batches
        print(f"Epoch {epoch + 1}/{EPOCHS} | BPR Loss: {avg_loss:.4f}")

    print("Training Complete!")
    # Save the trained matrices
    np.save("data/trained_P.npy", np.array(params['P']))
    np.save("data/trained_Q.npy", np.array(params['Q']))
    print("Saved optimized P and Q matrices to disk.")

if __name__ == "__main__":
    train_model()