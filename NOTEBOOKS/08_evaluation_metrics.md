# 08. Evaluation Metrics

Evaluate Popularity, Rating-based, Content-based, and Collaborative Filtering models on the test set.

---

### **1. Import Libraries**

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import load_npz
from collections import defaultdict
from tqdm.notebook import tqdm

sns.set_theme(style='whitegrid')
```

---

### **2. Load Data & Models**

```python
test_df = pd.read_csv('test_interactions.csv')
print(f'Test shape: {test_df.shape}')

# Get ground truth for each user
# Only consider items with rating >= 4 as positive interactions (hits)
positive_test = test_df[test_df['rating'] >= 4]
ground_truth = positive_test.groupby('authorid')['recipeid'].apply(set).to_dict()

# We will only evaluate users who have at least one positive interaction in test
eval_users = list(ground_truth.keys())
print(f'Total users to evaluate: {len(eval_users)}')
```

**Output:**
```
Test shape: (166906, 10)
Total users to evaluate: 26116
```

```python
# Optional: Sample a subset of users to speed up evaluation
import random
random.seed(42)
if len(eval_users) > 5000:
    eval_users = random.sample(eval_users, 5000)
    print(f'Sampled {len(eval_users)} users for faster evaluation.')
```

**Output:**
```
Sampled 5000 users for faster evaluation.
```

---

### **3. Evaluation Metrics Definitions**

```python
def precision_at_k(actual, predicted, k):
    if not actual:
        return 0.0
    pred_k = set(predicted[:k])
    hits = len(pred_k.intersection(actual))
    return hits / k

def recall_at_k(actual, predicted, k):
    if not actual:
        return 0.0
    pred_k = set(predicted[:k])
    hits = len(pred_k.intersection(actual))
    return hits / len(actual)

def hitrate_at_k(actual, predicted, k):
    if not actual:
        return 0.0
    pred_k = set(predicted[:k])
    return 1.0 if len(pred_k.intersection(actual)) > 0 else 0.0

def ndcg_at_k(actual, predicted, k):
    if not actual:
        return 0.0
    dcg = 0.0
    for i, p in enumerate(predicted[:k]):
        if p in actual:
            dcg += 1.0 / np.log2(i + 2)

    idcg = sum((1.0 / np.log2(i + 2)) for i in range(min(k, len(actual))))
    return dcg / idcg if idcg > 0 else 0.0
```

---

### **4. Generate Predictions & Compute Metrics**

All four models are evaluated: **Popularity**, **Rating-based**, **Content-based**, and **Collaborative Filtering (SVD)**.

```python
metrics = {'Model': [], 'Precision@5': [], 'Recall@5': [], 'HitRate@10': [], 'NDCG@10': []}

def evaluate_model(model_name, predictions_dict):
    p5, r5, hr10, ndcg10 = [], [], [], []
    for u in eval_users:
        actual = ground_truth.get(u, set())
        if not actual:
            continue
        preds = predictions_dict.get(u, [])

        p5.append(precision_at_k(actual, preds, 5))
        r5.append(recall_at_k(actual, preds, 5))
        hr10.append(hitrate_at_k(actual, preds, 10))
        ndcg10.append(ndcg_at_k(actual, preds, 10))

    metrics['Model'].append(model_name)
    metrics['Precision@5'].append(np.mean(p5))
    metrics['Recall@5'].append(np.mean(r5))
    metrics['HitRate@10'].append(np.mean(hr10))
    metrics['NDCG@10'].append(np.mean(ndcg10))
    print(f"{model_name} Evaluation Complete.")
```

**Output:**
```
Popularity Evaluation Complete.
Rating-based Evaluation Complete.
Content-based Evaluation Complete.
Collaborative filtering Evaluation Complete.
```

---

### **5. Final Results Table**

```python
results_df = pd.DataFrame(metrics)
display(results_df)
```

**Output Preview:**

| # | Model | Precision@5 | Recall@5 | HitRate@10 | NDCG@10 |
|---|-------|-------------|----------|------------|---------|
| 0 | Popularity | 0.00776 | 0.011431 | 0.0618 | 0.013920 |
| 1 | Rating-based | 0.00056 | 0.000334 | 0.0058 | 0.000884 |
| 2 | Content-based | 0.00224 | 0.005210 | 0.0184 | 0.005381 |
| 3 | Collaborative filtering | 0.00156 | 0.001858 | 0.0150 | 0.002524 |

---

### **6. Evaluation Metrics Comparison**

**Graph Generation Code:**
```python
metrics_to_plot = ['Precision@5', 'Recall@5', 'HitRate@10', 'NDCG@10']
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for i, metric in enumerate(metrics_to_plot):
    sns.barplot(data=results_df, x='Model', y=metric, ax=axes[i], palette='viridis')
    axes[i].set_title(metric)
    axes[i].set_xlabel('')
    axes[i].set_ylabel('')
    axes[i].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()
```

**Visualization:**
![Evaluation Metrics Comparison](output/plots/08_evaluation_metrics.png)
