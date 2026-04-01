# High-Performance E-Commerce Recommendation Engine

An end-to-end, GPU-accelerated recommendation microservice built from scratch using JAX and FastAPI.

This project implements a custom Bayesian Personalized Ranking (BPR) matrix factorization engine. By abandoning black-box libraries (like LightFM) and writing the linear algebra natively in JAX, the system achieves highly optimized training on NVIDIA GPUs via XLA compilation, and serves personalized recommendations in <35ms via vectorized NumPy inference.

## ✨ Key Engineering Highlights

*   **Mathematical Engine:** Custom BPR loss function implemented in pure JAX, optimizing latent item/user embeddings through lock-free Stochastic Gradient Descent (Optax/Adam).
*   **Memory Optimization:** Engineered a highly efficient data pipeline translating 100k+ sparse string UUIDs into contiguous integer arrays, compressing gigabytes of relational e-commerce tables into lightweight memory footprints.
*   **Low-Latency Inference:** Achieved sub-35ms API response times (single CPU core) by utilizing raw $O(1)$ memory lookups, vectorized dot products, and real-time metadata hydration.
*   **Novelty Filtering:** Post-processing layer mathematically filters historical purchases ($-\infty$ scoring) in real-time, preventing the "recommending the past" trap.

## 🛠 Tech Stack

*   **ML Engine:** JAX, Optax, NumPy, SciPy
*   **Data Engineering:** Pandas
*   **API / Serving:** FastAPI, Uvicorn
*   **Dataset:** Olist Brazilian E-Commerce Dataset (95k+ Users, 32k+ Items)

## 🏗 System Architecture

*   **Data Pipeline (`prep_data.py`):** Ingests relational CSVs, performs entity resolution, maps UUIDs to dense continuous integers, and exports raw math matrices.
*   **Metadata Hydration (`export_metadata.py`):** Extracts human-readable product categories and maps them identically to the integer matrices for $O(1)$ API lookups.
*   **Training Loop (`train.py`):** Uses JAX `@jax.jit` JIT compilation to calculate gradients. Dynamically generates negative samples and optimizes $P$ (User) and $Q$ (Item) matrices over $N$ epochs.
*   **API Gateway (`api.py`):** Loads optimized matrices into server RAM. Receives user ID, computes vector dot products against the entire 32k+ catalog, filters history, and returns top $K$ hydrated results.

## 🚀 Installation & Setup

### 1. Environment (WSL2 / Linux Recommended)

Ensure you have Python 3.10+ installed.

```bash
python -m venv venv
source venv/bin/activate
pip install pandas numpy fastapi uvicorn optax
```

### 2. Install JAX with NVIDIA GPU Support

```bash
pip install -U "jax[cuda12]"
```

### 3. Dataset

Download the Olist Dataset from Kaggle, extract the .zip, and place the CSVs inside a `data/` directory at the root of this project.

## 💻 Usage

Run the pipeline sequentially:

### Step 1: Compile the Data

```bash
python src/prep_data.py
```
Outputs: `data/jax_interactions.npy`

### Step 2: Export Metadata Dictionary

```bash
python src/export_metadata.py
```
Outputs: `data/item_metadata.json`

### Step 3: Train the Latent Embeddings (GPU Accelerated)

```bash
python src/train.py
```
Outputs: `data/trained_P.npy`, `data/trained_Q.npy`

### Step 4: Launch the Microservice

```bash
uvicorn src.api:app --reload
```

## 📡 API Reference

`GET /recommend/{user_idx}?top_k=10`

Fetches personalized product recommendations for a specific user.

**Example Response:**

```json
{
  "user_idx": 100,
  "compute_time_ms": 33.49,
  "filtered_items_count": 2,
  "recommendations": [
    {
      "internal_idx": 100,
      "product_id": "97f1396a5a1f7c07ba51784efdec44b8",
      "category": "utilidades_domesticas",
      "match_score": 0.0783
    },
    {
      "internal_idx": 3986,
      "product_id": "6dde44b4172999f35f08654d06bad633",
      "category": "moveis_escritorio",
      "match_score": 0.0461
    }
  ]
}
```
