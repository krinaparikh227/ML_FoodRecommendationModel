# 10. Multi-Objective Ranking

This notebook implements a fixed-weight multi-objective ranking algorithm, combining the SVD Preference Score, the newly generated Composite Health Score, and Preparation Time to recommend optimal food items.


```python
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import MinMaxScaler

os.makedirs('output', exist_ok=True)
```

## 1. Load Models and Features


```python
# Load SVD Model
with open('../RESULTS/WEEK 03/07/svd_model.pkl', 'rb') as f:
    svd_model = pickle.load(f)

# Load Features
features_df = pd.read_csv('../DATA/PROCESSED/food_metadata_with_health.csv')
# Rename columns to match expected schema
if 'id' in features_df.columns:
    features_df.rename(columns={'id': 'recipeid'}, inplace=True)
if 'name' in features_df.columns:
    features_df['recipe_name'] = features_df['name']
if 'minutes' in features_df.columns:
    features_df['prep_time'] = features_df['minutes']
if 'category' not in features_df.columns:
    features_df['category'] = 'Unknown'
if 'item_id' in features_df.columns:
    features_df.rename(columns={'item_id': 'recipeid'}, inplace=True)
features_df = features_df.drop_duplicates(subset=['recipeid']).set_index('recipeid')

print(f"Loaded {len(features_df)} feature records.")
```

    Loaded 40968 feature records.
    

## 2. Normalize Objective Scores
- **Preference:** Output of SVD is mapped approximately to `[0, 1]` using `(rating - 1) / 4`.
- **Health:** `health_score` is already composite and effectively distributed between `[0, 1]`.
- **Time:** Inversely normalized `prep_time`.


```python
scaler = MinMaxScaler()

features_df['health_score'].fillna(features_df['health_score'].median(), inplace=True)
features_df['prep_time'].fillna(features_df['prep_time'].median(), inplace=True)

# Time Normalization (Inverse: Faster = Higher Score)
prep_99 = features_df['prep_time'].quantile(0.99)
clipped_time = features_df['prep_time'].clip(0, prep_99)
features_df['time_score'] = 1.0 - scaler.fit_transform(clipped_time.values.reshape(-1, 1)).flatten()

item_metrics = features_df[['health_score', 'time_score', 'recipe_name', 'category']].to_dict(orient='index')
print("Item metrics ready.")
```

    C:\Users\Kush Shah\AppData\Local\Temp\ipykernel_19968\1710878239.py:3: ChainedAssignmentError: A value is being set on a copy of a DataFrame or Series through chained assignment using an inplace method.
    Such inplace method never works to update the original DataFrame or Series, because the intermediate object on which we are setting values always behaves as a copy (due to Copy-on-Write).
    
    For example, when doing 'df[col].method(value, inplace=True)', try using 'df.method({col: value}, inplace=True)' instead, to perform the operation inplace on the original object, or try to avoid an inplace operation using 'df[col] = df[col].method(value)'.
    
    See the documentation for a more detailed explanation: https://pandas.pydata.org/pandas-docs/stable/user_guide/copy_on_write.html
      features_df['health_score'].fillna(features_df['health_score'].median(), inplace=True)
    C:\Users\Kush Shah\AppData\Local\Temp\ipykernel_19968\1710878239.py:4: ChainedAssignmentError: A value is being set on a copy of a DataFrame or Series through chained assignment using an inplace method.
    Such inplace method never works to update the original DataFrame or Series, because the intermediate object on which we are setting values always behaves as a copy (due to Copy-on-Write).
    
    For example, when doing 'df[col].method(value, inplace=True)', try using 'df.method({col: value}, inplace=True)' instead, to perform the operation inplace on the original object, or try to avoid an inplace operation using 'df[col] = df[col].method(value)'.
    
    See the documentation for a more detailed explanation: https://pandas.pydata.org/pandas-docs/stable/user_guide/copy_on_write.html
      features_df['prep_time'].fillna(features_df['prep_time'].median(), inplace=True)
    

    Item metrics ready.
    

## 3. Implement Fixed-Weight Multi-Objective Scoring
Here we define our function and apply a balanced weight vector (e.g. `Pref=0.5, Health=0.3, Time=0.2`).


```python
def get_multi_objective_recommendations(user_id, candidate_items, w_pref=0.5, w_health=0.3, w_time=0.2, top_n=10):
    scored_candidates = []
    
    for item in candidate_items:
        # 1. Preference Score
        est_rating = svd_model.predict(user_id, item).est
        pref_score = (est_rating - 1.0) / 4.0
        
        # 2. Extract item metrics
        metrics = item_metrics.get(item, {'health_score': 0, 'time_score': 0, 'recipe_name': 'Unknown', 'category': 'Unknown'})
        
        # 3. Final Composite Score
        final_score = (w_pref * pref_score) + (w_health * metrics['health_score']) + (w_time * metrics['time_score'])
        
        scored_candidates.append({
            'recipeid': item,
            'recipe_name': metrics['recipe_name'],
            'category': metrics['category'],
            'pref_score': pref_score,
            'health_score': metrics['health_score'],
            'time_score': metrics['time_score'],
            'final_score': final_score
        })
        
    # Sort descending by final score
    scored_candidates.sort(key=lambda x: x['final_score'], reverse=True)
    
    return pd.DataFrame(scored_candidates[:top_n])
```

## 4. Test on a Sample User


```python
# Let's pick a user and run it on a small candidate set (to save time)
sample_user = 1533  # Example user ID, replace with a known ID if needed
import random
random.seed(42)
all_items = list(item_metrics.keys())
candidate_subset = random.sample(all_items, min(1000, len(all_items)))

recs_df = get_multi_objective_recommendations(sample_user, candidate_subset)
display(recs_df)

# Save to output
recs_df.to_csv('../RESULTS/WEEK 04/sample_multi_objective_recs.csv', index=False)
print("[SUCCESS] Saved sample recommendations to ../RESULTS/WEEK 04/sample_multi_objective_recs.csv")
# Augment metadata with normalized scores for downstream notebooks
print('Saving augmented metadata for Notebook 11 & 21...')
train_df = pd.read_csv('../DATA/PROCESSED/train_interactions.csv')
popularity_counts = train_df['recipe_id'].value_counts().to_dict()
features_df_save = features_df.reset_index()
if 'recipeid' in features_df_save.columns:
    features_df_save.rename(columns={'recipeid': 'id'}, inplace=True)
if 'recipe_name' in features_df_save.columns:
    features_df_save['name'] = features_df_save['recipe_name']
if 'prep_time' in features_df_save.columns:
    features_df_save['minutes'] = features_df_save['prep_time']
features_df_save['time_score_norm'] = features_df_save['time_score']
features_df_save['interaction_count'] = features_df_save['id'].map(lambda x: popularity_counts.get(x, 0))
features_df_save['pop_score'] = np.log1p(features_df_save['interaction_count'])
min_pop = features_df_save['pop_score'].min()
max_pop = features_df_save['pop_score'].max()
features_df_save['pop_score_norm'] = (features_df_save['pop_score'] - min_pop) / (max_pop - min_pop + 1e-9)
output_meta_path = '../DATA/PROCESSED/food_metadata_with_health.csv'
features_df_save.to_csv(output_meta_path, index=False)
print('Saved successfully to ' + output_meta_path)

```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>recipeid</th>
      <th>recipe_name</th>
      <th>category</th>
      <th>pref_score</th>
      <th>health_score</th>
      <th>time_score</th>
      <th>final_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3742</td>
      <td>pacific rim glazed flank steak</td>
      <td>Unknown</td>
      <td>0.984407</td>
      <td>0.718733</td>
      <td>1.000000</td>
      <td>0.907824</td>
    </tr>
    <tr>
      <th>1</th>
      <td>51650</td>
      <td>nogales steak tacos</td>
      <td>Unknown</td>
      <td>1.000000</td>
      <td>0.748482</td>
      <td>0.900990</td>
      <td>0.904743</td>
    </tr>
    <tr>
      <th>2</th>
      <td>69154</td>
      <td>spicy mussels in white wine sauce</td>
      <td>Unknown</td>
      <td>0.996313</td>
      <td>0.704843</td>
      <td>0.964356</td>
      <td>0.902480</td>
    </tr>
    <tr>
      <th>3</th>
      <td>282730</td>
      <td>creole&nbsp;&nbsp;style&nbsp;&nbsp;chicken&nbsp;&nbsp; sausage jambalaya</td>
      <td>Unknown</td>
      <td>1.000000</td>
      <td>0.745515</td>
      <td>0.891089</td>
      <td>0.901872</td>
    </tr>
    <tr>
      <th>4</th>
      <td>133511</td>
      <td>grilled pork chops with lime&nbsp;&nbsp;cilantro&nbsp;&nbsp; garlic</td>
      <td>Unknown</td>
      <td>0.992736</td>
      <td>0.702512</td>
      <td>0.960396</td>
      <td>0.899201</td>
    </tr>
    <tr>
      <th>5</th>
      <td>136443</td>
      <td>low fat creamy baked salmon</td>
      <td>Unknown</td>
      <td>0.954334</td>
      <td>0.777188</td>
      <td>0.940594</td>
      <td>0.898442</td>
    </tr>
    <tr>
      <th>6</th>
      <td>55242</td>
      <td>barbecued red roast pork tenderloin</td>
      <td>Unknown</td>
      <td>0.993051</td>
      <td>0.716199</td>
      <td>0.930693</td>
      <td>0.897524</td>
    </tr>
    <tr>
      <th>7</th>
      <td>339365</td>
      <td>baked horseradish salmon with chardonnay chive...</td>
      <td>Unknown</td>
      <td>0.987226</td>
      <td>0.710609</td>
      <td>0.950495</td>
      <td>0.896894</td>
    </tr>
    <tr>
      <th>8</th>
      <td>147438</td>
      <td>shrimp or crab louis</td>
      <td>Unknown</td>
      <td>0.993933</td>
      <td>0.692695</td>
      <td>0.960396</td>
      <td>0.896854</td>
    </tr>
    <tr>
      <th>9</th>
      <td>14174</td>
      <td>crispy fish in chili sauce</td>
      <td>Unknown</td>
      <td>0.973511</td>
      <td>0.750395</td>
      <td>0.920792</td>
      <td>0.896032</td>
    </tr>
  </tbody>
</table>
</div>


    [SUCCESS] Saved sample recommendations to output/sample_multi_objective_recs.csv
    Saving augmented metadata for Notebook 11 & 21...
    

    Saved successfully to ../DATA/PROCESSED/food_metadata_with_health.csv
    
