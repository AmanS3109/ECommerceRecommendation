import pandas as pd
import json

def export_item_metadata(data_dir="data/"):
    print("Loading raw Olist CSVs to build the translation dictionary...")
    
    # 1. Load the raw tables
    customers = pd.read_csv(f"{data_dir}olist_customers_dataset.csv")
    orders = pd.read_csv(f"{data_dir}olist_orders_dataset.csv")
    items = pd.read_csv(f"{data_dir}olist_order_items_dataset.csv")
    products = pd.read_csv(f"{data_dir}olist_products_dataset.csv")

    print("Rebuilding the exact interaction map from prep_data.py...")
    # Link Order -> Customer -> Unique Customer
    user_orders = pd.merge(
        orders[['order_id', 'customer_id']], 
        customers[['customer_id', 'customer_unique_id']], 
        on='customer_id'
    )
    
    # Link User directly to Product ID
    interactions = pd.merge(
        user_orders, 
        items[['order_id', 'product_id']], 
        on='order_id'
    )
    raw_df = interactions[['customer_unique_id', 'product_id']].dropna()

    # 2. Get the unique items in the EXACT SAME ORDER as we did for the JAX matrix
    unique_items = raw_df['product_id'].unique()
    item_to_idx = {item_str: idx for idx, item_str in enumerate(unique_items)}
    
    # 3. Create a lookup dictionary mapping product_id to category
    product_names = dict(zip(products['product_id'], products['product_category_name']))
    
    metadata_dict = {}
    print("Hydrating the dictionary...")
    for original_id, integer_idx in item_to_idx.items():
        readable_name = product_names.get(original_id, "Unknown Category")
        
        # Catch any NaN (Not a Number) values from Pandas and convert to string
        if pd.isna(readable_name):
            readable_name = "Unknown Category"
            
        metadata_dict[int(integer_idx)] = {
            "original_id": original_id,
            "product_name": str(readable_name)
        }
        
    # 4. Save to disk as JSON
    with open(f"{data_dir}item_metadata.json", "w") as f:
        json.dump(metadata_dict, f)
        
    print(f"Success! Exported metadata for {len(metadata_dict)} items.")

if __name__ == "__main__":
    export_item_metadata()