# 11 - Weight Ablation Analysis
**Project:** GroundedNutriRec
**Role:** LLM/RAG + Multi-Objective Ranking
**Task:** Evaluate multiple combinations of preference, health, popularity, diversity, and preparation time weights. Fill the comparison table.



```python

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

# Define paths
DATA_DIR = "d:/MainCodes/Internship/DATA/PROCESSED"
TRAIN_FILE = os.path.join(DATA_DIR, "train_interactions.csv")
TEST_FILE = os.path.join(DATA_DIR, "test_interactions.csv")
METADATA_FILE = os.path.join(DATA_DIR, "food_metadata_with_health.csv")
EMBEDDINGS_FILE = os.path.join(DATA_DIR, "recipe_embeddings.npy")

RESULTS_DIR_V4 = "d:/MainCodes/Internship/RESULTS/V4"
RESULTS_DIR_WEEK4 = "d:/MainCodes/Internship/RESULTS/WEEK 04"
os.makedirs(RESULTS_DIR_V4, exist_ok=True)
os.makedirs(RESULTS_DIR_WEEK4, exist_ok=True)

```

## 1. Load Processed Datasets
We load the interaction splits, augmented recipe metadata, and precomputed embeddings.



```python

print("Loading train/test interactions...")
train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)

print("Loading recipe metadata and embeddings...")
meta_df = pd.read_csv(METADATA_FILE)
recipe_embeddings = np.load(EMBEDDINGS_FILE)

# Prepare mappings
recipe_to_idx = {row['id']: idx for idx, row in meta_df.iterrows()}
idx_to_recipe = {idx: row['id'] for idx, row in meta_df.iterrows()}

train_history = train_df.groupby('user_id')['recipe_id'].apply(set).to_dict()
train_liked = train_df[train_df['liked'] == 1].groupby('user_id')['recipe_id'].apply(list).to_dict()

# Positives in test split
test_pos = test_df[test_df['liked'] == 1]
test_pos_dict = test_pos.groupby('user_id')['recipe_id'].apply(set).to_dict()
test_users = list(test_pos_dict.keys())

print(f"Loaded {len(test_users)} test users for evaluation.")

```

    Loading train/test interactions...
    

    Loading recipe metadata and embeddings...
    

    Loaded 17329 test users for evaluation.
    

## 2. Prepare User Profiles and Item Feature Vectors
We align the recipe metadata scores (health, popularity, preparation time) into fast 1D NumPy arrays.



```python

# Extract user profiles
user_profiles = np.zeros((len(test_users), 384), dtype=np.float32)
for u_idx, uid in enumerate(test_users):
    liked_ids = train_liked.get(uid, [])
    if not liked_ids:
        liked_ids = list(train_history.get(uid, []))
    if liked_ids:
        indices = [recipe_to_idx[rid] for rid in liked_ids if rid in recipe_to_idx]
        if indices:
            user_profiles[u_idx] = recipe_embeddings[indices].mean(axis=0)

# Extract item-level score components as NumPy vectors for fast broadcasting
health_scores = meta_df['health_score'].values.reshape(1, -1)
pop_scores = meta_df['pop_score_norm'].values.reshape(1, -1)
time_scores = meta_df['time_score_norm'].values.reshape(1, -1)

# Raw metrics for evaluation
raw_health = meta_df['health_score'].values
raw_minutes = meta_df['minutes'].values

print("Aligned profile matrices and item vectors.")

```

    Aligned profile matrices and item vectors.
    

## 3. Define Vectorized Multi-Objective Evaluation
To evaluate all 17,329 test users efficiently, we process recommendations in batches using vector operations.
For each batch of users, we:
1. Compute Content-Based similarity matrix.
2. Min-max normalize similarities to $[0, 1]$ row-wise.
3. Combine scores using weights.
4. Mask the user's training history.
5. Extract top-10 recommended recipe indices and evaluate metrics.



```python

def evaluate_variant(alpha, batch_size=1000):
    """
    evaluate_variant
    
    Evaluates a specific weight combination of the multi-objective ranking framework
    over the entire test user population in optimized vector batches.
    
    @param  {list} alpha       - Weight configuration: [w_pref, w_health, w_pop, w_div, w_time].
    @param  {int} batch_size   - Batch size of users to process in parallel (default 1000).
    @returns {dict}            - Dictionary of computed average metrics: Precision@10,
                                 Recall@10, Avg Health, Avg Prep Time, Diversity (ILD@10).
    @validates                 - Masking ensures that already seen recipes are excluded.
    @edge-cases                - Handles users with zero liking history gracefully.
    """
    precisions = []
    recalls = []
    healths = []
    prep_times = []
    diversities = []
    
    num_batches = int(np.ceil(len(test_users) / batch_size))
    
    for b in range(num_batches):
        start_idx = b * batch_size
        end_idx = min(start_idx + batch_size, len(test_users))
        batch_uids = test_users[start_idx:end_idx]
        batch_profiles = user_profiles[start_idx:end_idx]
        
        # 1. Content-based similarity (preference score)
        sims = cosine_similarity(batch_profiles, recipe_embeddings)
        
        # 2. Min-max normalize similarities per user
        sims_min = sims.min(axis=1, keepdims=True)
        sims_max = sims.max(axis=1, keepdims=True)
        pref_scores = (sims - sims_min) / (sims_max - sims_min + 1e-9)
        
        # 3. Diversity score is 1.0 - pref_score
        div_scores = 1.0 - pref_scores
        
        # 4. Weighted score aggregation
        final_scores = (
            alpha[0] * pref_scores +
            alpha[1] * health_scores +
            alpha[2] * pop_scores +
            alpha[3] * div_scores +
            alpha[4] * time_scores
        )
        
        # 5. Mask already seen interactions per user
        for idx, uid in enumerate(batch_uids):
            seen = train_history.get(uid, set())
            seen_indices = [recipe_to_idx[rid] for rid in seen if rid in recipe_to_idx]
            if seen_indices:
                final_scores[idx, seen_indices] = -np.inf
                
        # 6. Extract top-10 recommendations for each user
        top10_partition = np.argpartition(final_scores, -10, axis=1)[:, -10:]
        
        for idx, uid in enumerate(batch_uids):
            row_scores = final_scores[idx]
            row_partition = top10_partition[idx]
            # Sort top 10 descending
            top10_idx = row_partition[np.argsort(-row_scores[row_partition])]
            recs = [idx_to_recipe[i] for i in top10_idx]
            
            # Evaluate Accuracy
            positives = test_pos_dict[uid]
            hits = len(set(recs) & positives)
            precisions.append(hits / 10.0)
            recalls.append(hits / len(positives))
            
            # Evaluate Raw Health and Prep Time of Recommendations
            healths.append(raw_health[top10_idx].mean())
            prep_times.append(raw_minutes[top10_idx].mean())
            
            # Evaluate Intra-List Diversity (ILD) using embedding cosine distance
            recs_emb = recipe_embeddings[top10_idx]
            emb_sim = cosine_similarity(recs_emb)
            # ILD is 1 - average pairwise cosine similarity of top-10
            ild = 1.0 - (emb_sim[np.triu_indices(10, k=1)].sum() / 45.0)
            diversities.append(ild)
            
    return {
        'Precision@10': np.mean(precisions),
        'Recall@10': np.mean(recalls),
        'Avg Health Score': np.mean(healths),
        'Avg Prep Time': np.mean(prep_times),
        'Diversity': np.mean(diversities)
    }

```

## 4. Run Weight Ablation Study
We execute the evaluation for the four weight configurations.



```python

variants = {
    'Preference only': [1.0, 0.0, 0.0, 0.0, 0.0],
    'Preference + Health': [0.5, 0.5, 0.0, 0.0, 0.0],
    'Preference + Health + Time': [0.4, 0.4, 0.0, 0.0, 0.2],
    'Full Multi-Objective': [0.3, 0.3, 0.1, 0.1, 0.2]
}

results = []
for name, alpha in variants.items():
    print(f"Evaluating variant: {name} with weights {alpha}...")
    metrics = evaluate_variant(alpha)
    metrics['Variant'] = name
    results.append(metrics)

# Create DataFrame
results_df = pd.DataFrame(results)[['Variant', 'Precision@10', 'Recall@10', 'Avg Health Score', 'Avg Prep Time', 'Diversity']]
print("\n=== ABLATION STUDY RESULTS ===")
print(results_df.to_string(index=False))

```

    Evaluating variant: Preference only with weights [1.0, 0.0, 0.0, 0.0, 0.0]...
    

    Evaluating variant: Preference + Health with weights [0.5, 0.5, 0.0, 0.0, 0.0]...
    

    Evaluating variant: Preference + Health + Time with weights [0.4, 0.4, 0.0, 0.0, 0.2]...
    

    Evaluating variant: Full Multi-Objective with weights [0.3, 0.3, 0.1, 0.1, 0.2]...
    

    
    === ABLATION STUDY RESULTS ===
                       Variant  Precision@10  Recall@10  Avg Health Score  Avg Prep Time  Diversity
               Preference only      0.000629   0.001767          0.624353      68.341220   0.294158
           Preference + Health      0.000381   0.001108          0.748488      90.105205   0.334136
    Preference + Health + Time      0.000323   0.000953          0.738585      35.510970   0.335187
          Full Multi-Objective      0.003024   0.008576          0.708864      27.976883   0.402703
    

## 5. Plot and Save Ablation Visualizations
We create comparative charts showing how precision, health score, and preparation time trade off across variants.



```python

# Save results table
results_table_path = os.path.join(RESULTS_DIR_WEEK4, "ablation_results_table.csv")
results_df.to_csv(results_table_path, index=False)
results_df.to_csv(os.path.join(RESULTS_DIR_V4, "ablation_results_table.csv"), index=False)

# Bar charts of performance comparison
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
variants_names = results_df['Variant'].tolist()

# 1. Precision@10
axes[0, 0].bar(variants_names, results_df['Precision@10'], color='teal', alpha=0.8)
axes[0, 0].set_title('Precision@10')
axes[0, 0].tick_params(axis='x', rotation=15)

# 2. Avg Health Score
axes[0, 1].bar(variants_names, results_df['Avg Health Score'], color='forestgreen', alpha=0.8)
axes[0, 1].set_title('Average Health Score')
axes[0, 1].tick_params(axis='x', rotation=15)

# 3. Avg Prep Time (Minutes)
axes[1, 0].bar(variants_names, results_df['Avg Prep Time'], color='indianred', alpha=0.8)
axes[1, 0].set_title('Average Preparation Time (Minutes)')
axes[1, 0].tick_params(axis='x', rotation=15)

# 4. Diversity (ILD@10)
axes[1, 1].bar(variants_names, results_df['Diversity'], color='darkorchid', alpha=0.8)
axes[1, 1].set_title('Intra-List Diversity (ILD@10)')
axes[1, 1].tick_params(axis='x', rotation=15)

plt.tight_layout()
plot_path_week4 = os.path.join(RESULTS_DIR_WEEK4, "ablation_tradeoff_chart.png")
plot_path_v4 = os.path.join(RESULTS_DIR_V4, "ablation_tradeoff_chart.png")
plt.savefig(plot_path_week4, dpi=300)
plt.savefig(plot_path_v4, dpi=300)
plt.close()

print(f"Saved ablation trade-off charts to {plot_path_week4} and {plot_path_v4}.")
```

    Saved ablation trade-off charts to d:/MainCodes/Internship/RESULTS/WEEK 04\ablation_tradeoff_chart.png and d:/MainCodes/Internship/RESULTS/V4\ablation_tradeoff_chart.png.
    
