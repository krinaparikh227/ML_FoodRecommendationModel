# 21 - Pareto-Based Multi-Objective Ranking
**Project:** GroundedNutriRec
**Role:** LLM/RAG + Multi-Objective Ranking
**Task:** Implement Pareto dominance and non-dominated sorting to rank recipes. Compare Pareto-front ranking against fixed weighted scoring.



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

## 1. Load Datasets and Prep Mappings
Load the processed interaction files, recipe embeddings, and metadata containing the normalized objective scores.



```python

print("Loading train/test interactions...")
train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)

print("Loading recipe metadata and embeddings...")
meta_df = pd.read_csv(METADATA_FILE)
recipe_embeddings = np.load(EMBEDDINGS_FILE)

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
    

## 2. Load and Align User Profiles and Item Objectives
Align the recipe features so we can extract them quickly by index coordinates.



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

# Precomputed item-level scores
health_vec = meta_df['health_score'].values
pop_vec = meta_df['pop_score_norm'].values
time_vec = meta_df['time_score_norm'].values

raw_health = meta_df['health_score'].values
raw_minutes = meta_df['minutes'].values

```

## 3. Implement Fast Non-Dominated Sorting
We define a function to compute Pareto dominance.
To optimize execution, we do not rank all 41,240 recipes. For each user, we first form a candidate pool of the top 100 items from each objective, then execute non-dominated sorting over this pool.



```python

def fast_non_dominated_sort(objectives):
    """
    fast_non_dominated_sort
    
    Partitions a matrix of objective vectors into hierarchical Pareto fronts
    using vector-accelerated dominance evaluations.
    
    @param  {ndarray} objectives - Matrix of shape (N, 5) representing N items and 5 objectives.
    @returns {list}             - List of list of integers. Each sublist corresponds to a front
                                  containing indices of the items in that front.
    @validates                  - Validates that objectives array is non-empty.
    @edge-cases                 - Returns empty list if input shape is empty.
    """
    num_items = objectives.shape[0]
    if num_items == 0:
        return []
    
    # Vectorized dominance checks using broadcasting
    diff = objectives[:, np.newaxis, :] - objectives[np.newaxis, :, :]
    
    # p dominates q if:
    # 1. objectives[p] >= objectives[q] in all dimensions
    # 2. objectives[p] > objectives[q] in at least one dimension
    all_ge = np.all(diff >= 0, axis=2)
    any_gt = np.any(diff > 0, axis=2)
    dominates = all_ge & any_gt
    
    # n[q] is count of items dominating q (sum columns)
    n = dominates.sum(axis=0)
    
    # S[p] is list of items dominated by p
    S = [np.where(dominates[p])[0].tolist() for p in range(num_items)]
    
    fronts = [np.where(n == 0)[0].tolist()]
            
    i = 0
    while len(fronts[i]) > 0:
        next_front = []
        for p in fronts[i]:
            for q in S[p]:
                n[q] -= 1
                if n[q] == 0:
                    next_front.append(q)
        i += 1
        fronts.append(next_front)
        
    return fronts[:-1]

```

## 4. Run Pareto Evaluation
We evaluate the Pareto ranker on the test user population.



```python

def evaluate_pareto(batch_size=1000):
    """
    evaluate_pareto
    
    Generates and evaluates recommendations generated by non-dominated sorting over
    the candidate pool for the entire test user dataset.
    
    @param  {int} batch_size   - Number of users to process in parallel vector operations (default 1000).
    @returns {dict}            - Average performance metrics of the Pareto ranking variant.
    @validates                 - Filters out interacted recipes from candidates.
    @edge-cases                - Handles small front sizes by pooling consecutive fronts.
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
        
        # 1. Cosine similarity for preference
        sims = cosine_similarity(batch_profiles, recipe_embeddings)
        sims_min = sims.min(axis=1, keepdims=True)
        sims_max = sims.max(axis=1, keepdims=True)
        pref_scores = (sims - sims_min) / (sims_max - sims_min + 1e-9)
        div_scores = 1.0 - pref_scores
        
        # For each user in the batch, perform Pareto ranking on their candidate pool
        for idx, uid in enumerate(batch_uids):
            seen = train_history.get(uid, set())
            seen_indices = {recipe_to_idx[rid] for rid in seen if rid in recipe_to_idx}
            
            u_pref = pref_scores[idx]
            u_div = div_scores[idx]
            
            # Form candidate pool: top 80 of each objective
            top_pref = np.argsort(-u_pref)[:120]
            top_health = np.argsort(-health_vec)[:80]
            top_pop = np.argsort(-pop_vec)[:80]
            top_div = np.argsort(-u_div)[:80]
            top_time = np.argsort(-time_vec)[:80]
            
            pool = list(set(top_pref) | set(top_health) | set(top_pop) | set(top_div) | set(top_time))
            # Filter out seen
            pool = [i for i in pool if i not in seen_indices]
            
            # Collate objective scores: shape (len(pool), 5)
            pool_objectives = np.column_stack([
                u_pref[pool],
                health_vec[pool],
                pop_vec[pool],
                u_div[pool],
                time_vec[pool]
            ])
            
            # Run non-dominated sort
            fronts = fast_non_dominated_sort(pool_objectives)
            
            # Rank items by Front number ascending
            recs_idx = []
            for front in fronts:
                # Break ties within the front by their average normalized score
                front_avgs = pool_objectives[front].mean(axis=1)
                sorted_front_items = [pool[front[i]] for i in np.argsort(-front_avgs)]
                recs_idx.extend(sorted_front_items)
                if len(recs_idx) >= 10:
                    break
                    
            top10_idx = recs_idx[:10]
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
            
        print(f"Batch {b+1}/{num_batches} complete...")
            
    return {
        'Variant': 'Pareto-Front Ranking',
        'Precision@10': np.mean(precisions),
        'Recall@10': np.mean(recalls),
        'Avg Health Score': np.mean(healths),
        'Avg Prep Time': np.mean(prep_times),
        'Diversity': np.mean(diversities)
    }

```

## 5. Run Pareto Evaluation and Output Comparison
Run the evaluation and compare the results of Pareto ranking against the weighted scoring variants.



```python

print("Evaluating Pareto-Front Ranking...")
pareto_metrics = evaluate_pareto()
pareto_df = pd.DataFrame([pareto_metrics])

# Load ablation results to make comparison
ablation_results_path = os.path.join(RESULTS_DIR_V4, "ablation_results_table.csv")
if os.path.exists(ablation_results_path):
    ablation_df = pd.read_csv(ablation_results_path)
    comparison_df = pd.concat([ablation_df, pareto_df], ignore_index=True)
else:
    comparison_df = pareto_df

print("\n=== BASELINE COMPARISON TABLE (INCLUDING PARETO) ===")
print(comparison_df.to_string(index=False))

# Save results
comparison_df.to_csv(os.path.join(RESULTS_DIR_WEEK4, "comparison_results_with_pareto.csv"), index=False)
comparison_df.to_csv(os.path.join(RESULTS_DIR_V4, "comparison_results_with_pareto.csv"), index=False)

```

    Evaluating Pareto-Front Ranking...
    

    Batch 1/18 complete...
    

    Batch 2/18 complete...
    

    Batch 3/18 complete...
    

    Batch 4/18 complete...
    

    Batch 5/18 complete...
    

    Batch 6/18 complete...
    

    Batch 7/18 complete...
    

    Batch 8/18 complete...
    

    Batch 9/18 complete...
    

    Batch 10/18 complete...
    

    Batch 11/18 complete...
    

    Batch 12/18 complete...
    

    Batch 13/18 complete...
    

    Batch 14/18 complete...
    

    Batch 15/18 complete...
    

    Batch 16/18 complete...
    

    Batch 17/18 complete...
    

    Batch 18/18 complete...
    
    === BASELINE COMPARISON TABLE (INCLUDING PARETO) ===
                       Variant  Precision@10  Recall@10  Avg Health Score  Avg Prep Time  Diversity
               Preference only      0.000629   0.001767          0.624353      68.341220   0.294158
           Preference + Health      0.000381   0.001108          0.748488      90.105205   0.334136
    Preference + Health + Time      0.000323   0.000953          0.738585      35.510970   0.335187
          Full Multi-Objective      0.003024   0.008576          0.708864      27.976883   0.402703
          Pareto-Front Ranking      0.006671   0.017495          0.670971      41.726216   0.588329
    

## 6. Plot Pareto Comparison Chart
Plot comparative bar charts including the Pareto model.



```python

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
names = comparison_df['Variant'].tolist()

# 1. Precision@10
axes[0, 0].bar(names, comparison_df['Precision@10'], color=['teal']*4 + ['blue'], alpha=0.8)
axes[0, 0].set_title('Precision@10')
axes[0, 0].tick_params(axis='x', rotation=15)

# 2. Avg Health Score
axes[0, 1].bar(names, comparison_df['Avg Health Score'], color=['forestgreen']*4 + ['green'], alpha=0.8)
axes[0, 1].set_title('Average Health Score')
axes[0, 1].tick_params(axis='x', rotation=15)

# 3. Avg Prep Time (Minutes)
axes[1, 0].bar(names, comparison_df['Avg Prep Time'], color=['indianred']*4 + ['red'], alpha=0.8)
axes[1, 0].set_title('Average Preparation Time (Minutes)')
axes[1, 0].tick_params(axis='x', rotation=15)

# 4. Diversity (ILD@10)
axes[1, 1].bar(names, comparison_df['Diversity'], color=['darkorchid']*4 + ['magenta'], alpha=0.8)
axes[1, 1].set_title('Intra-List Diversity (ILD@10)')
axes[1, 1].tick_params(axis='x', rotation=15)

plt.tight_layout()
plot_path_week4 = os.path.join(RESULTS_DIR_WEEK4, "pareto_comparison_chart.png")
plot_path_v4 = os.path.join(RESULTS_DIR_V4, "pareto_comparison_chart.png")
plt.savefig(plot_path_week4, dpi=300)
plt.savefig(plot_path_v4, dpi=300)
plt.close()

print(f"Saved Pareto comparison charts to {plot_path_week4} and {plot_path_v4}.")
```

    Saved Pareto comparison charts to d:/MainCodes/Internship/RESULTS/WEEK 04\pareto_comparison_chart.png and d:/MainCodes/Internship/RESULTS/V4\pareto_comparison_chart.png.
    
