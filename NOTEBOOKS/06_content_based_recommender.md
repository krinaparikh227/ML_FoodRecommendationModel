# 06. Content-Based Recommender

This document implements a Content-Based Recommender using TF-IDF on recipe name, ingredients, and category.

---

### **1. Import Libraries**

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import pickle
import re

sns.set_theme(style='whitegrid')
```

---

### **2. Load Data**

```python
features_path = 'food_features_engineered.csv'
features_df = pd.read_csv(features_path)
features_df['item_id'] = features_df['item_id'].astype(str)
print(f'Features shape: {features_df.shape}')
```

**Output:**
```
Features shape: (522517, 16)
```

---

### **3. Text Preprocessing**

Combine the textual features to form a document for each item.

```python
def preprocess_text(row):
    name = str(row.get('recipe_name', ''))
    ingredients = str(row.get('ingredients', ''))
    # Clean the ingredients string e.g., c("sugar", "flour") -> sugar flour
    ingredients = re.sub(r'c\(|\)|\"|,', ' ', ingredients)
    category = str(row.get('category', ''))

    # Combine textual features
    combined = f"{name} {ingredients} {category}"
    # Basic cleaning
    combined = combined.lower()
    combined = re.sub(r'[^a-z ]', ' ', combined)
    combined = re.sub(r'\s+', ' ', combined).strip()
    return combined

features_df['content_text'] = features_df.apply(preprocess_text, axis=1)
print('Sample preprocessed text:')
print(features_df['content_text'].head())
```

**Output:**
```
Sample preprocessed text:
0    low fat berry blue frozen dessert blueberries ...
1    biryani saffron milk hot green chili peppers o...
2    best lemonade sugar lemons rind of lemon zest ...
3    carina s tofu vegetable kebabs extra firm tofu...
4    cabbage soup plain tomato juice cabbage onion ...
Name: content_text, dtype: str
```

---

### **4. TF-IDF Vectorization**

```python
# Using a subset of items to avoid memory explosion if the dataset is huge.
num_items = 10000
subset_df = features_df.head(num_items).reset_index(drop=True)

tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
tfidf_matrix = tfidf.fit_transform(subset_df['content_text'])

print('TF-IDF matrix shape:', tfidf_matrix.shape)
```

**Output:**
```
TF-IDF matrix shape: (10000, 4899)
```

---

### **5. Compute Similarities & Predict**

For a given test user, we can find items similar to what they have liked in the training set. To simplify for evaluation, we compute cosine similarity for a batch of items.

**Similarity Computation Code:**
```python
# Example: Find similar items for item at index 0
item_idx = 0
cosine_sim = cosine_similarity(tfidf_matrix[item_idx], tfidf_matrix).flatten()
similar_indices = cosine_sim.argsort()[:-10:-1]  # top 9 most similar

print(f"\nItems similar to {subset_df.iloc[item_idx]['recipe_name']}:")
for idx in similar_indices:
    if idx != item_idx:
        print(f"{subset_df.iloc[idx]['recipe_name']} (score: {cosine_sim[idx]:.3f})")
```

**Output:**
```
Items similar to Low-Fat Berry Blue Frozen Dessert:
Frozen Yogurt-On-A-Stick (score: 0.510)
Blueberry Dream Cake (score: 0.508)
Caramel Apple Milkshakes (score: 0.468)
Very Berry Parfaits (score: 0.452)
Blueberry Apples (score: 0.432)
Frozen Lemon Souffle (score: 0.429)
Cranberry & Pineapple Frozen Dessert (score: 0.424)
Low-Fat Berry Good Smoothie (score: 0.423)
```

**Graph Generation Code:**
```python
target_recipe = subset_df.iloc[item_idx]['recipe_name']

names = []
scores = []
for idx in similar_indices:
    if idx != item_idx:
        names.append(subset_df.iloc[idx]['recipe_name'])
        scores.append(cosine_sim[idx])

plt.figure(figsize=(10, 6))
sns.barplot(x=scores, y=names, palette='Purples_r')
plt.title(f'Cosine Similarity to "{target_recipe}"')
plt.xlabel('Cosine Similarity Score')
plt.ylabel('')
plt.tight_layout()
plt.show()
```

**Visualization:**
![Cosine Similarity Chart](output/plots/06_content_similarity.png)

---

### **6. Save Model / Process for Evaluation**

We save the subset dataset and the TF-IDF matrix to disk so the evaluation notebook can use it to generate recommendations for test users.

```python
os.makedirs('output', exist_ok=True)

with open('output/content_items.pkl', 'wb') as f:
    pickle.dump(subset_df['item_id'].tolist(), f)

from scipy.sparse import save_npz
save_npz('output/tfidf_matrix.npz', tfidf_matrix)
print('Saved content-based model artifacts.')
```

**Output:**
```
Saved content-based model artifacts.
```
