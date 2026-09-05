# 07 - Collaborative Filtering Recommender Models

**Project:** GroundedNutriRec  
**Role:** Baseline Recommendation + Evaluation Lead  
**Scope:** Implement and evaluate three collaborative filtering recommendation baselines on the Food.com dataset:
1. **User-Item Cosine Similarity**: User-based KNN Collaborative Filtering using sparse matrix operations.
2. **Surprise SVD**: Matrix factorization using the `scikit-surprise` library.
3. **Implicit ALS**: Alternating Least Squares factorization using the `implicit` library.

## Objectives:
1. Load preprocessed train and test interaction datasets (`train_interactions.csv`, `test_interactions.csv`).
2. Map user and recipe IDs to contiguous zero-indexed integers.
3. Implement ranking metrics: Precision@K, Recall@K, HitRate@K, NDCG@K, and MRR@K.
4. Train and evaluate the three Collaborative Filtering approaches.
5. Print a comparative summary table of all models for $K \in \{5, 10\}$.

## 1. Setup and Imports


```python
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Dataset, Reader, SVD
from implicit.als import AlternatingLeastSquares
import time

print("Setup complete. Libraries imported successfully.")
```

    Setup complete. Libraries imported successfully.
    

## 2. Load Datasets


```python
print("Loading datasets...")
train_df = pd.read_csv("../DATA/PROCESSED/train_interactions.csv")
test_df = pd.read_csv("../DATA/PROCESSED/test_interactions.csv")
recipes_df = pd.read_csv("../DATA/RAW/RAW_recipes.csv")

print(f"Train interactions: {len(train_df):,}")
print(f"Test interactions: {len(test_df):,}")
print(f"Recipes count: {len(recipes_df):,}")
```

    Loading datasets...
    

    Train interactions: 437,558
    Test interactions: 118,060
    Recipes count: 231,637
    

## 3. Id/Index Mappings


```python
# Create mappings
all_users = sorted(list(set(train_df['user_id'].unique()) | set(test_df['user_id'].unique())))
all_recipes = sorted(list(set(train_df['recipe_id'].unique()) | set(test_df['recipe_id'].unique())))

user_to_idx = {uid: idx for idx, uid in enumerate(all_users)}
recipe_to_idx = {rid: idx for idx, rid in enumerate(all_recipes)}

idx_to_user = {idx: uid for uid, idx in user_to_idx.items()}
idx_to_recipe = {idx: rid for rid, idx in recipe_to_idx.items()}

num_users = len(all_users)
num_recipes = len(all_recipes)
print(f"Unique Users: {num_users}, Unique Recipes: {num_recipes}")
```

    Unique Users: 17813, Unique Recipes: 41240
    

## 4. Prepare Evaluation Targets
For ranking evaluations, we target positive test interactions (defined by `liked = 1` in the test set). We also group each user's training history to filter out seen items during recommendation.


```python
# Filter test set for positive interactions (liked = 1)
test_pos = test_df[test_df['liked'] == 1]
# Group test set targets
test_targets = test_pos.groupby('user_id')['recipe_id'].apply(list).to_dict()
test_users = list(test_targets.keys())
print(f"Number of test users with positive interactions: {len(test_users):,}")

# Group train set history to filter out already seen items
train_history = train_df.groupby('user_id')['recipe_id'].apply(set).to_dict()
```

    Number of test users with positive interactions: 17,329
    

## 5. Implement Evaluation Metrics
Here we define top-K recommendation metrics: Precision@K, Recall@K, HitRate@K, NDCG@K, and MRR@K.


```python
def evaluate_model(recommendations_dict, k_list=[5, 10]):
    metrics = {k: {'precision': [], 'recall': [], 'hitrate': [], 'ndcg': [], 'mrr': []} for k in k_list}
    
    for uid in test_users:
        actual = test_targets[uid]
        recs = recommendations_dict.get(uid, [])
        
        for k in k_list:
            recs_k = recs[:k]
            hits = len(set(recs_k) & set(actual))
            
            # Precision
            precision = hits / k
            
            # Recall
            recall = hits / len(actual) if len(actual) > 0 else 0.0
            
            # Hit Rate
            hitrate = 1.0 if hits > 0 else 0.0
            
            # NDCG
            dcg = 0.0
            for i, p in enumerate(recs_k):
                if p in actual:
                    dcg += 1.0 / np.log2(i + 2)
            idcg = sum([1.0 / np.log2(i + 2) for i in range(min(k, len(actual)))])
            ndcg = dcg / idcg if idcg > 0.0 else 0.0
            
            # MRR
            mrr = 0.0
            for i, p in enumerate(recs_k):
                if p in actual:
                    mrr = 1.0 / (i + 1)
                    break
            
            metrics[k]['precision'].append(precision)
            metrics[k]['recall'].append(recall)
            metrics[k]['hitrate'].append(hitrate)
            metrics[k]['ndcg'].append(ndcg)
            metrics[k]['mrr'].append(mrr)
            
    results = {}
    for k in k_list:
        results[k] = {metric: np.mean(values) for metric, values in metrics[k].items()}
    return results
```

## 6. User-Item Matrix with Cosine Similarity (User-Based KNN CF)
We build a sparse user-item matrix using explicit ratings. We compute user cosine similarities, filter them to retain only the top 100 neighbors for each user, and use sparse matrix multiplication to compute item recommendation scores.


```python
print("Constructing User-Item rating matrix...")
rows = train_df['user_id'].map(user_to_idx).values
cols = train_df['recipe_id'].map(recipe_to_idx).values
vals = train_df['rating'].values

rating_matrix = csr_matrix((vals, (rows, cols)), shape=(num_users, num_recipes))

print("Computing user cosine similarity...")
start_time = time.time()
user_sim = cosine_similarity(rating_matrix, dense_output=True)
np.fill_diagonal(user_sim, 0) # Remove self-similarity

print("Sparsifying similarity matrix (top K=100 nearest neighbors)...")
k_neighbors = 100
for i in range(num_users):
    row = user_sim[i]
    threshold = np.partition(row, -k_neighbors)[-k_neighbors]
    row[row < threshold] = 0.0

user_sim_sparse = csr_matrix(user_sim)
print(f"Sparse similarity matrix shape: {user_sim_sparse.shape}, non-zero entries: {user_sim_sparse.nnz:,}")

print("Generating User-Based Cosine CF recommendations...")
cosine_recs = {}
chunk_size = 2000
for start_idx in range(0, num_users, chunk_size):
    end_idx = min(start_idx + chunk_size, num_users)
    chunk_scores = user_sim_sparse[start_idx:end_idx].dot(rating_matrix).toarray()
    
    for i in range(start_idx, end_idx):
        uid = idx_to_user[i]
        if uid in test_targets:
            seen = train_history.get(uid, set())
            scores = chunk_scores[i - start_idx]
            for r in seen:
                if r in recipe_to_idx:
                    scores[recipe_to_idx[r]] = -np.inf
            top_indices = np.argsort(scores)[::-1][:10]
            cosine_recs[uid] = [idx_to_recipe[idx] for idx in top_indices]

print(f"Cosine CF recommendations generated in {time.time() - start_time:.2f} seconds.")
```

    Constructing User-Item rating matrix...
    Computing user cosine similarity...
    

    Sparsifying similarity matrix (top K=100 nearest neighbors)...
    

    Sparse similarity matrix shape: (17813, 17813), non-zero entries: 1,723,472
    Generating User-Based Cosine CF recommendations...
    

    Cosine CF recommendations generated in 36.02 seconds.
    

## 7. Surprise SVD (Matrix Factorization)
We use the `scikit-surprise` library to perform SVD matrix factorization. We train on the explicit ratings and predict preferences for unobserved user-recipe pairs.


```python
print("Preparing data for Surprise...")
start_time = time.time()
# Load data from Pandas DataFrame
reader = Reader(rating_scale=(1, 5))
surprise_data = Dataset.load_from_df(train_df[['user_id', 'recipe_id', 'rating']], reader)

trainset = surprise_data.build_full_trainset()

print("Training SVD model...")
# Using standard latent factors (n_factors=20) and epochs (n_epochs=15)
svd_model = SVD(n_factors=20, n_epochs=15, random_state=42)
svd_model.fit(trainset)
print("SVD model training completed.")

print("Aligning SVD latent factor vectors with index mappings...")
n_factors = svd_model.n_factors
pu_aligned = np.zeros((num_users, n_factors))
bu_aligned = np.zeros(num_users)
for idx, uid in idx_to_user.items():
    try:
        inner_uid = trainset.to_inner_uid(uid)
        pu_aligned[idx] = svd_model.pu[inner_uid]
        bu_aligned[idx] = svd_model.bu[inner_uid]
    except ValueError:
        pass

qi_aligned = np.zeros((num_recipes, n_factors))
bi_aligned = np.zeros(num_recipes)
for idx, rid in idx_to_recipe.items():
    try:
        inner_iid = trainset.to_inner_iid(rid)
        qi_aligned[idx] = svd_model.qi[inner_iid]
        bi_aligned[idx] = svd_model.bi[inner_iid]
    except ValueError:
        pass

global_mean = trainset.global_mean

print("Generating Surprise SVD recommendations (vectorized)...")
svd_recs = {} 
chunk_size = 2000
for start_idx in range(0, num_users, chunk_size):
    end_idx = min(start_idx + chunk_size, num_users)
    chunk_scores = (
        global_mean +
        bu_aligned[start_idx:end_idx, np.newaxis] +
        bi_aligned[np.newaxis, :] +
        np.dot(pu_aligned[start_idx:end_idx], qi_aligned.T)
    )
    for i in range(start_idx, end_idx):
        uid = idx_to_user[i]
        if uid in test_targets:
            seen = train_history.get(uid, set())
            scores = chunk_scores[i - start_idx]
            for r in seen:
                if r in recipe_to_idx:
                    scores[recipe_to_idx[r]] = -np.inf
            top_indices = np.argsort(scores)[::-1][:10]
            svd_recs[uid] = [idx_to_recipe[idx] for idx in top_indices]

print(f"Surprise SVD recommendations generated in {time.time() - start_time:.2f} seconds.")
# Save the trained SVD model for use in multi-objective ranking
import pickle
import os
os.makedirs('../RESULTS/WEEK 03/07', exist_ok=True)
with open('../RESULTS/WEEK 03/07/svd_model.pkl', 'wb') as f:
    pickle.dump(svd_model, f)
print('SVD model saved successfully to ../RESULTS/WEEK 03/07/svd_model.pkl')

```

    Preparing data for Surprise...
    

    Training SVD model...
    

    SVD model training completed.
    Aligning SVD latent factor vectors with index mappings...
    Generating Surprise SVD recommendations (vectorized)...
    

    Surprise SVD recommendations generated in 34.80 seconds.
    

    SVD model saved successfully to output/svd_model.pkl
    

## 8. Implicit ALS (Alternating Least Squares)
We use the `implicit` library to train an Alternating Least Squares (ALS) model on the binary implicit feedback labels (`liked`). We use the library's built-in batch recommendation for ultra-fast generation.


```python
print("Preparing data for Implicit ALS...")
start_time = time.time()
# Use liked as feedback weight
# Filter out rows where liked is NaN (unrated reviews)
train_implicit = train_df[train_df['liked'].notna()].copy()
train_implicit['liked'] = train_implicit['liked'].astype(np.float32)

# Row indices correspond to users, column indices to recipes
als_rows = train_implicit['user_id'].map(user_to_idx).values
als_cols = train_implicit['recipe_id'].map(recipe_to_idx).values
als_vals = train_implicit['liked'].values

user_items_als = csr_matrix((als_vals, (als_rows, als_cols)), shape=(num_users, num_recipes))

print("Training Alternating Least Squares (ALS) model...")
als_model = AlternatingLeastSquares(factors=64, regularization=0.05, iterations=15, random_state=42)
als_model.fit(user_items_als)
print("ALS model training completed.")

print("Generating Implicit ALS recommendations...")
als_recs = {}
# Batch recommend for all test users using implicit's native fast recommend method
test_user_indices = [user_to_idx[uid] for uid in test_users]
# recommend method signature: recommend(userids, user_items, N, filter_already_liked_items=True)
ids_batch, scores_batch = als_model.recommend(
    test_user_indices,
    user_items_als[test_user_indices],
    N=10,
    filter_already_liked_items=True
)

for idx, uid in enumerate(test_users):
    als_recs[uid] = [idx_to_recipe[item_idx] for item_idx in ids_batch[idx]]

print(f"Implicit ALS recommendations generated in {time.time() - start_time:.2f} seconds.")
```

    Preparing data for Implicit ALS...
    Training Alternating Least Squares (ALS) model...
    

    C:\Users\Kush Shah\AppData\Local\Programs\Python\Python313\Lib\site-packages\implicit\cpu\als.py:96: RuntimeWarning: OpenBLAS is configured to use 18 threads. It is highly recommended to disable its internal threadpool by setting the environment variable 'OPENBLAS_NUM_THREADS=1' or by calling 'threadpoolctl.threadpool_limits(1, "blas")'. Having OpenBLAS use a threadpool can lead to severe performance issues here.
      check_blas_config()
    


      0%|          | 0/15 [00:00<?, ?it/s]


    ALS model training completed.
    Generating Implicit ALS recommendations...
    

    Implicit ALS recommendations generated in 5.76 seconds.
    

## 9. Perform Evaluation


```python
print("Evaluating Cosine Similarity CF...")
cosine_results = evaluate_model(cosine_recs)

print("Evaluating Surprise SVD...")
svd_results = evaluate_model(svd_recs)

print("Evaluating Implicit ALS...")
als_results = evaluate_model(als_recs)
```

    Evaluating Cosine Similarity CF...
    

    Evaluating Surprise SVD...
    

    Evaluating Implicit ALS...
    

## 10. Final Results Table


```python
print("="*108)
print(f"{'Model':<25} | {'Prec@5':<8} | {'Recall@5':<8} | {'Prec@10':<8} | {'Recall@10':<8} | {'HR@10':<8} | {'NDCG@10':<8} | {'MRR@10':<8}")
print("="*108)
print(f"{'Cosine CF (KNN)':<25} | {cosine_results[5]['precision']:.6f} | {cosine_results[5]['recall']:.6f} | {cosine_results[10]['precision']:.6f} | {cosine_results[10]['recall']:.6f} | {cosine_results[10]['hitrate']:.6f} | {cosine_results[10]['ndcg']:.6f} | {cosine_results[10]['mrr']:.6f}")
print(f"{'Surprise SVD':<25} | {svd_results[5]['precision']:.6f} | {svd_results[5]['recall']:.6f} | {svd_results[10]['precision']:.6f} | {svd_results[10]['recall']:.6f} | {svd_results[10]['hitrate']:.6f} | {svd_results[10]['ndcg']:.6f} | {svd_results[10]['mrr']:.6f}")
print(f"{'Implicit ALS':<25} | {als_results[5]['precision']:.6f} | {als_results[5]['recall']:.6f} | {als_results[10]['precision']:.6f} | {als_results[10]['recall']:.6f} | {als_results[10]['hitrate']:.6f} | {als_results[10]['ndcg']:.6f} | {als_results[10]['mrr']:.6f}")
print("="*108)
```

    ============================================================================================================
    Model                     | Prec@5   | Recall@5 | Prec@10  | Recall@10 | HR@10    | NDCG@10  | MRR@10  
    ============================================================================================================
    Cosine CF (KNN)           | 0.009822 | 0.012854 | 0.007935 | 0.021033 | 0.072249 | 0.016536 | 0.028099
    Surprise SVD              | 0.000462 | 0.000467 | 0.000479 | 0.000917 | 0.004732 | 0.000789 | 0.001522
    Implicit ALS              | 0.008160 | 0.010373 | 0.006879 | 0.017089 | 0.063247 | 0.013686 | 0.023355
    ============================================================================================================
    
