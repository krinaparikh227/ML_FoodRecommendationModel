# 07. Collaborative Filtering Recommender

This document implements a Collaborative Filtering model using SVD from the `Surprise` library.

---

### **1. Import Libraries**

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from surprise import Dataset, Reader, SVD
import pickle
import os

sns.set_theme(style='whitegrid')
```

---

### **2. Load Data**

```python
train_path = 'train_interactions.csv'
train_df = pd.read_csv(train_path)
print(f'Train shape: {train_df.shape}')

# We only need authorid, recipeid, and rating
train_df = train_df[['authorid', 'recipeid', 'rating']]
train_df.dropna(inplace=True)
print(f'Train shape after dropping NaNs: {train_df.shape}')
```

**Output:**
```
Train shape: (616473, 10)
Train shape after dropping NaNs: (616473, 3)
```

---

### **3. Rating Distribution Visualization**

**Graph Generation Code:**
```python
plt.figure(figsize=(8, 5))
sns.countplot(x='rating', data=train_df, palette='Oranges_d')
plt.title('Distribution of Ratings in Training Set')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.tight_layout()
plt.show()
```

**Visualization:**
![Rating Distribution](output/plots/07_rating_distribution.png)

---

### **4. Prepare Data for Surprise**

The `Reader` parses the file containing the ratings. We specify the `rating_scale`.

```python
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(train_df[['authorid', 'recipeid', 'rating']], reader)
trainset = data.build_full_trainset()
```

---

### **5. Train SVD Model**

```python
algo = SVD(n_factors=50, random_state=42)
print('Training SVD model...')
algo.fit(trainset)
print('Training completed.')
```

**Output:**
```
Training SVD model...
Training completed.
```

---

### **6. Save Model for Evaluation**

```python
os.makedirs('output', exist_ok=True)

with open('output/svd_model.pkl', 'wb') as f:
    pickle.dump(algo, f)

print('Saved SVD model to output/svd_model.pkl')
```

**Output:**
```
Saved SVD model to output/svd_model.pkl
```
