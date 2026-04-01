import pandas as pd
import numpy as np

def build_jax_dataset(data_dir="data/"):
    print("1. Loading raw Olist CSVs...")
    customers = pd.read_csv(f"{data_dir}olist_customers_dataset.csv")
    orders = pd.read_csv(f"{data_dir}olist_orders_dataset.csv")
    items = pd.read_csv(f"{data_dir}olist_order_items_dataset.csv")

    print("2. Joining tables to find who bought what...")
    # Link Order -> Customer String ID -> Unique Customer String ID
    user_orders = pd.merge(
        orders[['order_id', 'customer_id']], 
        customers[['customer_id', 'customer_unique_id']], 
        on='customer_id'
    )
    
    # Link the User directly to the Product ID
    interactions = pd.merge(
        user_orders, 
        items[['order_id', 'product_id']], 
        on='order_id'
    )
    
    # Keep only the two columns we care about
    raw_df = interactions[['customer_unique_id', 'product_id']].dropna()
    
    print("3. Building the Integer Dictionaries (The Core Translation)...")
    # Get unique lists of strings
    unique_users = raw_df['customer_unique_id'].unique()
    unique_items = raw_df['product_id'].unique()
    
    # Create dictionaries: String -> Integer (0 to N-1)
    user_to_idx = {user_str: idx for idx, user_str in enumerate(unique_users)}
    item_to_idx = {item_str: idx for idx, item_str in enumerate(unique_items)}
    
    print("4. Mapping the dataframe to Math-Ready Integers...")
    # Apply the dictionaries to our dataframe
    raw_df['user_idx'] = raw_df['customer_unique_id'].map(user_to_idx)
    raw_df['item_idx'] = raw_df['product_id'].map(item_to_idx)
    
    # Extract as a pure NumPy array for JAX
    # Shape will be [Number of Purchases, 2]
    jax_interactions = raw_df[['user_idx', 'item_idx']].values
    
    print(f"Success! Total Users (m): {len(unique_users)}")
    print(f"Success! Total Items (n): {len(unique_items)}")
    print(f"Success! Total Interactions: {len(jax_interactions)}")
    
    # Save the numpy array so we don't have to rebuild it every time
    np.save(f"{data_dir}jax_interactions.npy", jax_interactions)
    print("Saved to data/jax_interactions.npy")

if __name__ == "__main__":
    build_jax_dataset()