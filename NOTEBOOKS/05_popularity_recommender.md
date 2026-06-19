# 05. Popularity and Rating Based Recommenders

This document implements non-personalized recommenders based on item popularity (interaction count) and average rating.

---

### **1. Import Libraries**

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set_theme(style='whitegrid')
```

---

### **2. Load Data**

```python
data_dir = '.'
train_path = os.path.join(data_dir, 'train_interactions.csv')
features_path = os.path.join(data_dir, 'food_features_engineered.csv')

print('Loading data...')
train_df = pd.read_csv(train_path)
features_df = pd.read_csv(features_path)
print(f'Train shape: {train_df.shape}')
print(f'Features shape: {features_df.shape}')
```

**Output:**
```
Loading data...
Train shape: (616473, 10)
Features shape: (522517, 16)
```

---

### **3. Popularity-Based Recommender**

Recommend the most interacted food items.

**Feature Computation Code:**
```python
# Count interactions per recipe
popularity_df = train_df.groupby('recipeid').size().reset_index(name='interaction_count')
popularity_df = popularity_df.sort_values('interaction_count', ascending=False)

# Merge with recipe names
popularity_df = popularity_df.merge(
    features_df[['item_id', 'recipe_name']],
    left_on='recipeid', right_on='item_id', how='left'
)

print('Top 10 Popular Recipes:')
display(popularity_df.head(10))
```

**Output Preview:**

| # | recipeid | interaction_count | recipe_name |
|---|----------|-------------------|-------------|
| 0 | 45809 | 1560 | Bourbon Chicken |
| 1 | 27208 | 956 | To Die for Crock Pot Roast |
| 2 | 89204 | 861 | Crock-Pot Chicken With Black Beans & Cream Cheese |
| 3 | 39087 | 732 | Creamy Cajun Chicken Pasta |
| 4 | 32204 | 707 | "Whatever Floats Your Boat" Brownies! |
| 5 | 22782 | 653 | Jo Mama's World Famous Spaghetti |
| 6 | 25690 | 592 | Pancakes |
| 7 | 69173 | 590 | Kittencal's Italian Melt-In-Your-Mouth Meatballs |
| 8 | 54257 | 579 | Yes, Virginia There is a Great Meatloaf |
| 9 | 68955 | 575 | Japanese Mum's Chicken |

**Graph Generation Code:**
```python
plt.figure(figsize=(10, 6))
sns.barplot(data=popularity_df.head(10), x='interaction_count', y='recipe_name', palette='Blues_r')
plt.title('Top 10 Popular Recipes by Interaction Count')
plt.xlabel('Interaction Count')
plt.ylabel('')
plt.tight_layout()
plt.show()
```

**Visualization:**
![Top 10 Popular Recipes](output/plots/05_popularity_top10.png)

---

### **4. Rating-Based Recommender**

Recommend the highest-rated items that meet a minimum interaction threshold.

**Feature Computation Code:**
```python
min_interactions = 50

# Calculate average rating and interaction count
rating_df = train_df.groupby('recipeid').agg(
    avg_rating=('rating', 'mean'),
    interaction_count=('rating', 'count')
).reset_index()

# Filter by threshold
rating_based_df = rating_df[rating_df['interaction_count'] >= min_interactions]
rating_based_df = rating_based_df.sort_values('avg_rating', ascending=False)

# Merge with recipe names
rating_based_df = rating_based_df.merge(
    features_df[['item_id', 'recipe_name']],
    left_on='recipeid', right_on='item_id', how='left'
)

print(f'Top 10 Highest Rated Recipes (Min {min_interactions} interactions):')
display(rating_based_df.head(10))
```

**Output Preview:**

| # | recipeid | avg_rating | interaction_count | recipe_name |
|---|----------|-----------|-------------------|-------------|
| 0 | 87689 | 5.000000 | 54 | Cake Flour Substitute |
| 1 | 25094 | 4.966102 | 59 | My Chicken Parmigiana |
| 2 | 73348 | 4.962963 | 54 | Turkey Chowder |
| 3 | 107440 | 4.961538 | 52 | Oven Cooked Bacon With Black Pepper and Brown Sugar |
| 4 | 46365 | 4.956522 | 92 | Sangria |
| 5 | 13228 | 4.955224 | 67 | Honey Mustard |
| 6 | 4075 | 4.953125 | 64 | Chocolate Mint Candy (Fudge) |
| 7 | 42976 | 4.952381 | 84 | Brown Sugar Bundt Cake |
| 8 | 10840 | 4.949153 | 59 | Dry Rub for Barbecued Ribs |
| 9 | 63621 | 4.943396 | 53 | Mango Salsa #1 |

**Graph Generation Code:**
```python
plt.figure(figsize=(10, 6))
sns.barplot(data=rating_based_df.head(10), x='avg_rating', y='recipe_name', palette='Greens_r')
plt.xlim(4.9, 5.01)
plt.title(f'Top 10 Highest Rated Recipes (Min {min_interactions} interactions)')
plt.xlabel('Average Rating')
plt.ylabel('')
plt.tight_layout()
plt.show()
```

**Visualization:**
![Top 10 Highest Rated Recipes](output/plots/05_rating_top10.png)

---

### **5. Save Predictions for Evaluation**

```python
import pickle

# Top 1000 items should be plenty for evaluation up to K=10
pop_recs = popularity_df['recipeid'].head(1000).tolist()
rating_recs = rating_based_df['recipeid'].head(1000).tolist()

os.makedirs('output', exist_ok=True)
with open('output/pop_recs.pkl', 'wb') as f:
    pickle.dump(pop_recs, f)

with open('output/rating_recs.pkl', 'wb') as f:
    pickle.dump(rating_recs, f)

print('Saved popularity and rating recommendations to output/')
```

**Output:**
```
Saved popularity and rating recommendations to output/
```
