# 09 - Health Score Generation

**Project:** GroundedNutriRec  
**Scope:** Generate a composite Health Score for each recipe using normalized nutritional features (protein, calories, fat) and ingredient diversity.  

## Objectives:
1. Load recipe data from `RAW_recipes.csv` (archive_3 dataset).
2. Parse the `nutrition` field to extract calories, total fat, and protein values.
3. Parse the `ingredients` field to compute ingredient diversity (count of unique ingredients).
4. Normalize protein, calorie, fat, and ingredient-diversity features using Min-Max scaling.
5. Compute a composite **Health Score** by combining the normalized features with domain-informed weights.
6. Analyze the distribution of the Health Score across recipes.
7. Save the enriched recipe dataset with Health Scores for downstream use.

## 1. Setup and Imports


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import ast
import warnings

warnings.filterwarnings('ignore')

# Configure plotting aesthetics
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)

# Paths
recipes_path = Path("../DATA/PROCESSED/food_metadata_clean.csv")
output_path = Path("../DATA/PROCESSED/food_metadata_with_health.csv")

print(f"Source data: {recipes_path}")
print(f"Output destination: {output_path}")
print("Setup complete.")
```

    Source data: ..\DATA\PROCESSED\food_metadata_clean.csv
    Output destination: ..\DATA\PROCESSED\food_metadata_with_health.csv
    Setup complete.
    

## 2. Load Dataset


```python
print("Loading RAW_recipes.csv...")
df = pd.read_csv(recipes_path)
print(f"Dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"\nMissing values:")
print(df.isnull().sum())
df.head()
```

    Loading RAW_recipes.csv...
    

    Dataset shape: (40968, 25)
    Columns: ['name', 'id', 'minutes', 'contributor_id', 'submitted', 'tags', 'nutrition', 'n_steps', 'steps', 'description', 'ingredients', 'n_ingredients', 'calories', 'total_fat_pdv', 'sugar_pdv', 'sodium_pdv', 'protein_pdv', 'saturated_fat_pdv', 'carbohydrates_pdv', 'calorie_level', 'protein_level', 'fat_level', 'ingredient_count', 'preparation_complexity', 'health_score']
    
    Missing values:
    name                        0
    id                          0
    minutes                     0
    contributor_id              0
    submitted                   0
    tags                        0
    nutrition                   0
    n_steps                     0
    steps                       0
    description               823
    ingredients                 0
    n_ingredients               0
    calories                    0
    total_fat_pdv               0
    sugar_pdv                   0
    sodium_pdv                  0
    protein_pdv                 0
    saturated_fat_pdv           0
    carbohydrates_pdv           0
    calorie_level               0
    protein_level               0
    fat_level                   0
    ingredient_count            0
    preparation_complexity      0
    health_score                0
    dtype: int64
    




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
      <th>name</th>
      <th>id</th>
      <th>minutes</th>
      <th>contributor_id</th>
      <th>submitted</th>
      <th>tags</th>
      <th>nutrition</th>
      <th>n_steps</th>
      <th>steps</th>
      <th>description</th>
      <th>...</th>
      <th>sodium_pdv</th>
      <th>protein_pdv</th>
      <th>saturated_fat_pdv</th>
      <th>carbohydrates_pdv</th>
      <th>calorie_level</th>
      <th>protein_level</th>
      <th>fat_level</th>
      <th>ingredient_count</th>
      <th>preparation_complexity</th>
      <th>health_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>chicken lickin&nbsp;&nbsp;good&nbsp;&nbsp;pork chops</td>
      <td>63986</td>
      <td>500</td>
      <td>14664</td>
      <td>2003-06-06</td>
      <td>['weeknight', 'time-to-make', 'course', 'main-...</td>
      <td>[105.7, 8.0, 0.0, 26.0, 5.0, 4.0, 3.0]</td>
      <td>5</td>
      <td>['dredge pork chops in mixture of flour , salt...</td>
      <td>here's and old standby i enjoy from time to ti...</td>
      <td>...</td>
      <td>26.0</td>
      <td>5.0</td>
      <td>4.0</td>
      <td>3.0</td>
      <td>low</td>
      <td>low</td>
      <td>low</td>
      <td>7</td>
      <td>high</td>
      <td>7.321097</td>
    </tr>
    <tr>
      <th>1</th>
      <td>chile rellenos</td>
      <td>43026</td>
      <td>45</td>
      <td>52268</td>
      <td>2002-10-14</td>
      <td>['60-minutes-or-less', 'time-to-make', 'course...</td>
      <td>[94.0, 10.0, 0.0, 11.0, 11.0, 21.0, 0.0]</td>
      <td>9</td>
      <td>['drain green chiles', 'sprinkle cornstarch on...</td>
      <td>a favorite from a local restaurant no longer i...</td>
      <td>...</td>
      <td>11.0</td>
      <td>11.0</td>
      <td>21.0</td>
      <td>0.0</td>
      <td>low</td>
      <td>medium</td>
      <td>low</td>
      <td>5</td>
      <td>medium</td>
      <td>5.806383</td>
    </tr>
    <tr>
      <th>2</th>
      <td>chinese&nbsp;&nbsp;candy</td>
      <td>23933</td>
      <td>15</td>
      <td>35268</td>
      <td>2002-03-29</td>
      <td>['15-minutes-or-less', 'time-to-make', 'course...</td>
      <td>[232.7, 21.0, 77.0, 4.0, 6.0, 38.0, 8.0]</td>
      <td>4</td>
      <td>['melt butterscotch chips in heavy saucepan ov...</td>
      <td>a little different, and oh so good. i include ...</td>
      <td>...</td>
      <td>4.0</td>
      <td>6.0</td>
      <td>38.0</td>
      <td>8.0</td>
      <td>low</td>
      <td>low</td>
      <td>medium</td>
      <td>3</td>
      <td>low</td>
      <td>4.180232</td>
    </tr>
    <tr>
      <th>3</th>
      <td>grilled&nbsp;&nbsp;venison burgers</td>
      <td>54100</td>
      <td>26</td>
      <td>68357</td>
      <td>2003-02-15</td>
      <td>['30-minutes-or-less', 'time-to-make', 'course...</td>
      <td>[190.9, 10.0, 10.0, 10.0, 45.0, 15.0, 2.0]</td>
      <td>13</td>
      <td>['in bowl , mix dry ingredients', 'add venison...</td>
      <td>delicious venison burgers with that</td>
      <td>...</td>
      <td>10.0</td>
      <td>45.0</td>
      <td>15.0</td>
      <td>2.0</td>
      <td>low</td>
      <td>high</td>
      <td>low</td>
      <td>10</td>
      <td>high</td>
      <td>9.383604</td>
    </tr>
    <tr>
      <th>4</th>
      <td>healthy for them&nbsp;&nbsp;yogurt popsicles</td>
      <td>67664</td>
      <td>10</td>
      <td>91970</td>
      <td>2003-07-26</td>
      <td>['15-minutes-or-less', 'time-to-make', 'course...</td>
      <td>[164.6, 3.0, 5.0, 1.0, 4.0, 6.0, 11.0]</td>
      <td>3</td>
      <td>['mix all the ingredients using a blender', 'p...</td>
      <td>my children and their friends ask for my homem...</td>
      <td>...</td>
      <td>1.0</td>
      <td>4.0</td>
      <td>6.0</td>
      <td>11.0</td>
      <td>low</td>
      <td>low</td>
      <td>low</td>
      <td>3</td>
      <td>low</td>
      <td>8.972053</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 25 columns</p>
</div>



## 3. Parse Nutrition Field

The `nutrition` column in `RAW_recipes.csv` is stored as a string representation of a list:  
`[calories, total_fat (% DV), sugar (% DV), sodium (% DV), protein (% DV), saturated_fat (% DV), carbohydrates (% DV)]`

We will extract:
- **Calories** (index 0) — absolute kcal
- **Total Fat** (index 1) — % daily value
- **Protein** (index 4) — % daily value


```python
print("Parsing nutrition field...")

# Vectorized extraction using str.strip and str.split (much faster than ast.literal_eval)
nutrition_clean = df['nutrition'].str.strip('[]')
nutrition_split = nutrition_clean.str.split(',', expand=True).astype(float)

# Assign columns: [calories, total_fat, sugar, sodium, protein, saturated_fat, carbohydrates]
df['calories'] = nutrition_split[0]
df['total_fat_pdv'] = nutrition_split[1]
df['sugar_pdv'] = nutrition_split[2]
df['sodium_pdv'] = nutrition_split[3]
df['protein_pdv'] = nutrition_split[4]
df['saturated_fat_pdv'] = nutrition_split[5]
df['carbohydrates_pdv'] = nutrition_split[6]

print(f"\nNutrition features extracted. Sample:")
df[['name', 'calories', 'total_fat_pdv', 'protein_pdv']].head(10)
```

    Parsing nutrition field...
    
    Nutrition features extracted. Sample:
    




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
      <th>name</th>
      <th>calories</th>
      <th>total_fat_pdv</th>
      <th>protein_pdv</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>chicken lickin&nbsp;&nbsp;good&nbsp;&nbsp;pork chops</td>
      <td>105.7</td>
      <td>8.0</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>chile rellenos</td>
      <td>94.0</td>
      <td>10.0</td>
      <td>11.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>chinese&nbsp;&nbsp;candy</td>
      <td>232.7</td>
      <td>21.0</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>grilled&nbsp;&nbsp;venison burgers</td>
      <td>190.9</td>
      <td>10.0</td>
      <td>45.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>healthy for them&nbsp;&nbsp;yogurt popsicles</td>
      <td>164.6</td>
      <td>3.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>how i got my family to eat spinach&nbsp;&nbsp;spinach ca...</td>
      <td>166.1</td>
      <td>16.0</td>
      <td>19.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>i stole the idea from mirj&nbsp;&nbsp;sesame noodles</td>
      <td>783.4</td>
      <td>46.0</td>
      <td>36.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>immoral&nbsp;&nbsp;sandwich filling&nbsp;&nbsp;loose meat</td>
      <td>223.2</td>
      <td>22.0</td>
      <td>35.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>jiffy&nbsp;&nbsp;extra moist carrot cake</td>
      <td>612.1</td>
      <td>49.0</td>
      <td>15.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>land of nod&nbsp;&nbsp;cinnamon buns</td>
      <td>575.3</td>
      <td>18.0</td>
      <td>28.0</td>
    </tr>
  </tbody>
</table>
</div>



## 4. Compute Ingredient Diversity

Ingredient diversity is measured as the number of unique ingredients in each recipe (`n_ingredients` column). Recipes with a wider variety of ingredients tend to offer more diverse nutritional profiles.


```python
# Use the existing n_ingredients column directly (avoids slow ast.literal_eval parsing)
df['ingredient_diversity'] = df['n_ingredients'].copy()

print("Ingredient diversity statistics:")
print(df['ingredient_diversity'].describe())

df[['name', 'n_ingredients', 'ingredient_diversity']].head(10)
```

    Ingredient diversity statistics:
    count    40968.000000
    mean         8.739113
    std          3.632923
    min          1.000000
    25%          6.000000
    50%          8.000000
    75%         11.000000
    max         43.000000
    Name: ingredient_diversity, dtype: float64
    




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
      <th>name</th>
      <th>n_ingredients</th>
      <th>ingredient_diversity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>chicken lickin&nbsp;&nbsp;good&nbsp;&nbsp;pork chops</td>
      <td>7</td>
      <td>7</td>
    </tr>
    <tr>
      <th>1</th>
      <td>chile rellenos</td>
      <td>5</td>
      <td>5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>chinese&nbsp;&nbsp;candy</td>
      <td>3</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>grilled&nbsp;&nbsp;venison burgers</td>
      <td>10</td>
      <td>10</td>
    </tr>
    <tr>
      <th>4</th>
      <td>healthy for them&nbsp;&nbsp;yogurt popsicles</td>
      <td>3</td>
      <td>3</td>
    </tr>
    <tr>
      <th>5</th>
      <td>how i got my family to eat spinach&nbsp;&nbsp;spinach ca...</td>
      <td>8</td>
      <td>8</td>
    </tr>
    <tr>
      <th>6</th>
      <td>i stole the idea from mirj&nbsp;&nbsp;sesame noodles</td>
      <td>8</td>
      <td>8</td>
    </tr>
    <tr>
      <th>7</th>
      <td>immoral&nbsp;&nbsp;sandwich filling&nbsp;&nbsp;loose meat</td>
      <td>8</td>
      <td>8</td>
    </tr>
    <tr>
      <th>8</th>
      <td>jiffy&nbsp;&nbsp;extra moist carrot cake</td>
      <td>11</td>
      <td>11</td>
    </tr>
    <tr>
      <th>9</th>
      <td>land of nod&nbsp;&nbsp;cinnamon buns</td>
      <td>6</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
</div>



## 5. Exploratory Analysis of Nutritional Features

Before normalization, let's understand the distributions and identify outliers in our target features.


```python
features = ['calories', 'total_fat_pdv', 'protein_pdv', 'ingredient_diversity']

print("=" * 60)
print("NUTRITIONAL FEATURE STATISTICS (Raw)")
print("=" * 60)
print(df[features].describe())

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Distribution of Raw Nutritional Features', fontsize=16, fontweight='bold')

colors = ['#2ecc71', '#e74c3c', '#3498db', '#f39c12']
titles = ['Calories (kcal)', 'Total Fat (% DV)', 'Protein (% DV)', 'Ingredient Diversity']

for i, (feature, color, title) in enumerate(zip(features, colors, titles)):
    ax = axes[i // 2][i % 2]
    data = df[feature].dropna()
    
    # Clip for visualization (extreme outliers can distort histograms)
    clip_upper = data.quantile(0.99)
    data_clipped = data.clip(upper=clip_upper)
    
    ax.hist(data_clipped, bins=50, color=color, alpha=0.7, edgecolor='white')
    ax.set_title(title, fontsize=13)
    ax.set_xlabel(title)
    ax.set_ylabel('Count')
    ax.axvline(data.median(), color='black', linestyle='--', linewidth=1.5, label=f'Median: {data.median():.1f}')
    ax.legend(fontsize=10)

plt.tight_layout()
plt.show()
```

    ============================================================
    NUTRITIONAL FEATURE STATISTICS (Raw)
    ============================================================
               calories  total_fat_pdv   protein_pdv  ingredient_diversity
    count  40968.000000   40968.000000  40968.000000          40968.000000
    mean     437.915100      32.938122     33.725225              8.739113
    std      685.273339      64.736101     63.626597              3.632923
    min        0.000000       0.000000      0.000000              1.000000
    25%      168.300000       8.000000      6.000000              6.000000
    50%      296.700000      19.000000     18.000000              8.000000
    75%      487.325000      38.000000     50.000000             11.000000
    max    38680.100000    4331.000000   6552.000000             43.000000
    


    
## 6. Outlier Capping

Nutritional data can have extreme outliers (e.g., recipes with absurdly high calories). We cap at the 99th percentile to prevent outliers from dominating the normalization.


```python
print("Capping outliers at 99th percentile...\n")

for feature in features:
    q99 = df[feature].quantile(0.99)
    q01 = df[feature].quantile(0.01)
    
    outliers_above = (df[feature] > q99).sum()
    outliers_below = (df[feature] < q01).sum()
    
    print(f"{feature}:")
    print(f"  99th percentile: {q99:.2f}")
    print(f"  1st percentile:  {q01:.2f}")
    print(f"  Values capped above: {outliers_above}")
    print(f"  Values capped below: {outliers_below}")
    
    df[f'{feature}_capped'] = df[feature].clip(lower=q01, upper=q99)

print("\nCapped feature statistics:")
capped_features = [f'{f}_capped' for f in features]
print(df[capped_features].describe())
```

    Capping outliers at 99th percentile...
    
    calories:
      99th percentile: 3104.49
      1st percentile:  18.20
      Values capped above: 410
      Values capped below: 405
    total_fat_pdv:
      99th percentile: 264.00
      1st percentile:  0.00
      Values capped above: 408
      Values capped below: 0
    protein_pdv:
      99th percentile: 175.00
      1st percentile:  0.00
      Values capped above: 409
      Values capped below: 0
    ingredient_diversity:
      99th percentile: 19.00
      1st percentile:  2.00
      Values capped above: 351
      Values capped below: 13
    
    Capped feature statistics:
           calories_capped  total_fat_pdv_capped  protein_pdv_capped  \
    count     40968.000000          40968.000000        40968.000000   
    mean        418.009842             30.994874           32.480839   
    std         461.157570             40.780219           35.736941   
    min          18.200000              0.000000            0.000000   
    25%         168.300000              8.000000            6.000000   
    50%         296.700000             19.000000           18.000000   
    75%         487.325000             38.000000           50.000000   
    max        3104.490000            264.000000          175.000000   
    
           ingredient_diversity_capped  
    count                 40968.000000  
    mean                      8.712751  
    std                       3.533483  
    min                       2.000000  
    25%                       6.000000  
    50%                       8.000000  
    75%                      11.000000  
    max                      19.000000  
    

## 7. Min-Max Normalization

Normalize all features to [0, 1] range using Min-Max scaling:  
$x_{norm} = \frac{x - x_{min}}{x_{max} - x_{min}}$


```python
print("Applying Min-Max normalization...\n")

normalized_features = []
for feature in features:
    capped_col = f'{feature}_capped'
    norm_col = f'{feature}_norm'
    
    x_min = df[capped_col].min()
    x_max = df[capped_col].max()
    
    if x_max - x_min == 0:
        df[norm_col] = 0.0
    else:
        df[norm_col] = (df[capped_col] - x_min) / (x_max - x_min)
    
    normalized_features.append(norm_col)
    print(f"{feature}: min={x_min:.2f}, max={x_max:.2f} -> normalized to [0, 1]")

print("\nNormalized feature statistics:")
print(df[normalized_features].describe())

df[['name'] + normalized_features].head(10)
```

    Applying Min-Max normalization...
    
    calories: min=18.20, max=3104.49 -> normalized to [0, 1]
    total_fat_pdv: min=0.00, max=264.00 -> normalized to [0, 1]
    protein_pdv: min=0.00, max=175.00 -> normalized to [0, 1]
    ingredient_diversity: min=2.00, max=19.00 -> normalized to [0, 1]
    
    Normalized feature statistics:
           calories_norm  total_fat_pdv_norm  protein_pdv_norm  \
    count   40968.000000        40968.000000      40968.000000   
    mean        0.129544            0.117405          0.185605   
    std         0.149421            0.154471          0.204211   
    min         0.000000            0.000000          0.000000   
    25%         0.048634            0.030303          0.034286   
    50%         0.090238            0.071970          0.102857   
    75%         0.152003            0.143939          0.285714   
    max         1.000000            1.000000          1.000000   
    
           ingredient_diversity_norm  
    count               40968.000000  
    mean                    0.394868  
    std                     0.207852  
    min                     0.000000  
    25%                     0.235294  
    50%                     0.352941  
    75%                     0.529412  
    max                     1.000000  
    




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
      <th>name</th>
      <th>calories_norm</th>
      <th>total_fat_pdv_norm</th>
      <th>protein_pdv_norm</th>
      <th>ingredient_diversity_norm</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>chicken lickin&nbsp;&nbsp;good&nbsp;&nbsp;pork chops</td>
      <td>0.028351</td>
      <td>0.030303</td>
      <td>0.028571</td>
      <td>0.294118</td>
    </tr>
    <tr>
      <th>1</th>
      <td>chile rellenos</td>
      <td>0.024560</td>
      <td>0.037879</td>
      <td>0.062857</td>
      <td>0.176471</td>
    </tr>
    <tr>
      <th>2</th>
      <td>chinese&nbsp;&nbsp;candy</td>
      <td>0.069501</td>
      <td>0.079545</td>
      <td>0.034286</td>
      <td>0.058824</td>
    </tr>
    <tr>
      <th>3</th>
      <td>grilled&nbsp;&nbsp;venison burgers</td>
      <td>0.055957</td>
      <td>0.037879</td>
      <td>0.257143</td>
      <td>0.470588</td>
    </tr>
    <tr>
      <th>4</th>
      <td>healthy for them&nbsp;&nbsp;yogurt popsicles</td>
      <td>0.047436</td>
      <td>0.011364</td>
      <td>0.022857</td>
      <td>0.058824</td>
    </tr>
    <tr>
      <th>5</th>
      <td>how i got my family to eat spinach&nbsp;&nbsp;spinach ca...</td>
      <td>0.047922</td>
      <td>0.060606</td>
      <td>0.108571</td>
      <td>0.352941</td>
    </tr>
    <tr>
      <th>6</th>
      <td>i stole the idea from mirj&nbsp;&nbsp;sesame noodles</td>
      <td>0.247935</td>
      <td>0.174242</td>
      <td>0.205714</td>
      <td>0.352941</td>
    </tr>
    <tr>
      <th>7</th>
      <td>immoral&nbsp;&nbsp;sandwich filling&nbsp;&nbsp;loose meat</td>
      <td>0.066423</td>
      <td>0.083333</td>
      <td>0.200000</td>
      <td>0.352941</td>
    </tr>
    <tr>
      <th>8</th>
      <td>jiffy&nbsp;&nbsp;extra moist carrot cake</td>
      <td>0.192432</td>
      <td>0.185606</td>
      <td>0.085714</td>
      <td>0.529412</td>
    </tr>
    <tr>
      <th>9</th>
      <td>land of nod&nbsp;&nbsp;cinnamon buns</td>
      <td>0.180508</td>
      <td>0.068182</td>
      <td>0.160000</td>
      <td>0.235294</td>
    </tr>
  </tbody>
</table>
</div>



## 8. Normalized Feature Distributions


```python
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Distribution of Normalized Nutritional Features', fontsize=16, fontweight='bold')

norm_names = ['Calories (norm)', 'Total Fat (norm)', 'Protein (norm)', 'Ingredient Diversity (norm)']
colors = ['#2ecc71', '#e74c3c', '#3498db', '#f39c12']

for i, (col, name, color) in enumerate(zip(normalized_features, norm_names, colors)):
    ax = axes[i // 2][i % 2]
    data = df[col].dropna()
    ax.hist(data, bins=50, color=color, alpha=0.7, edgecolor='white')
    ax.set_title(name, fontsize=13)
    ax.set_xlabel('Normalized Value')
    ax.set_ylabel('Count')
    ax.axvline(data.median(), color='black', linestyle='--', linewidth=1.5, label=f'Median: {data.median():.3f}')
    ax.legend(fontsize=10)

plt.tight_layout()
plt.show()
```


    
## 9. Feature Correlation Analysis


```python
corr_df = df[normalized_features].rename(columns={
    'calories_norm': 'Calories',
    'total_fat_pdv_norm': 'Total Fat',
    'protein_pdv_norm': 'Protein',
    'ingredient_diversity_norm': 'Ingredient Diversity'
})

corr_matrix = corr_df.corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='RdYlBu_r', center=0, 
            fmt='.3f', linewidths=1, square=True,
            cbar_kws={'label': 'Correlation Coefficient'})
plt.title('Correlation Between Normalized Nutritional Features', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

print("Correlation matrix:")
print(corr_matrix)
```


    
Correlation matrix:
                          Calories  Total Fat   Protein  Ingredient Diversity
    Calories              1.000000   0.875172  0.617067              0.136348
    Total Fat             0.875172   1.000000  0.549545              0.123901
    Protein               0.617067   0.549545  1.000000              0.238534
    Ingredient Diversity  0.136348   0.123901  0.238534              1.000000
    

## 10. Health Score Computation

The Health Score is a weighted composite of the normalized features. The scoring logic is:

- **Higher protein** -> healthier (positive contribution)
- **Lower calories** -> healthier (inverted: 1 - normalized_calories)
- **Lower fat** -> healthier (inverted: 1 - normalized_fat)
- **Higher ingredient diversity** -> healthier (positive contribution, as diverse ingredients correlate with balanced nutrition)

### Weight Assignment:
| Feature | Weight | Rationale |
|---------|--------|-----------|
| Protein (norm) | 0.30 | High protein is a key health indicator |
| 1 - Calories (norm) | 0.30 | Lower calorie density promotes health |
| 1 - Fat (norm) | 0.25 | Lower fat content is generally healthier |
| Ingredient Diversity (norm) | 0.15 | Diverse ingredients support nutritional balance |

$$\text{Health Score} = 0.30 \times \text{protein}_{norm} + 0.30 \times (1 - \text{calories}_{norm}) + 0.25 \times (1 - \text{fat}_{norm}) + 0.15 \times \text{diversity}_{norm}$$


```python
# Define weights
W_PROTEIN = 0.30
W_CALORIES = 0.30
W_FAT = 0.25
W_DIVERSITY = 0.15

print("Computing Health Score...")
print(f"\nWeights:")
print(f"  Protein (positive):              {W_PROTEIN}")
print(f"  Calories (inverted):             {W_CALORIES}")
print(f"  Fat (inverted):                  {W_FAT}")
print(f"  Ingredient Diversity (positive): {W_DIVERSITY}")
print(f"  Total:                           {W_PROTEIN + W_CALORIES + W_FAT + W_DIVERSITY}")

# Compute Health Score
df['health_score'] = (
    W_PROTEIN * df['protein_pdv_norm'] +
    W_CALORIES * (1 - df['calories_norm']) +
    W_FAT * (1 - df['total_fat_pdv_norm']) +
    W_DIVERSITY * df['ingredient_diversity_norm']
)

# Handle NaN values (recipes with missing nutrition data)
nan_count = df['health_score'].isna().sum()
print(f"\nRecipes with NaN Health Score (missing nutrition data): {nan_count}")

print(f"\nHealth Score statistics:")
print(df['health_score'].describe())

df[['name', 'calories', 'total_fat_pdv', 'protein_pdv', 'ingredient_diversity', 'health_score']].head(15)
```

    Computing Health Score...
    
    Weights:
      Protein (positive):              0.3
      Calories (inverted):             0.3
      Fat (inverted):                  0.25
      Ingredient Diversity (positive): 0.15
      Total:                           1.0
    
    Recipes with NaN Health Score (missing nutrition data): 0
    
    Health Score statistics:
    count    40968.000000
    mean         0.596697
    std          0.074150
    min          0.012000
    25%          0.566115
    50%          0.596251
    75%          0.635774
    max          0.888066
    Name: health_score, dtype: float64
    




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
      <th>name</th>
      <th>calories</th>
      <th>total_fat_pdv</th>
      <th>protein_pdv</th>
      <th>ingredient_diversity</th>
      <th>health_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>chicken lickin&nbsp;&nbsp;good&nbsp;&nbsp;pork chops</td>
      <td>105.7</td>
      <td>8.0</td>
      <td>5.0</td>
      <td>7</td>
      <td>0.586608</td>
    </tr>
    <tr>
      <th>1</th>
      <td>chile rellenos</td>
      <td>94.0</td>
      <td>10.0</td>
      <td>11.0</td>
      <td>5</td>
      <td>0.578490</td>
    </tr>
    <tr>
      <th>2</th>
      <td>chinese&nbsp;&nbsp;candy</td>
      <td>232.7</td>
      <td>21.0</td>
      <td>6.0</td>
      <td>3</td>
      <td>0.528373</td>
    </tr>
    <tr>
      <th>3</th>
      <td>grilled&nbsp;&nbsp;venison burgers</td>
      <td>190.9</td>
      <td>10.0</td>
      <td>45.0</td>
      <td>10</td>
      <td>0.671474</td>
    </tr>
    <tr>
      <th>4</th>
      <td>healthy for them&nbsp;&nbsp;yogurt popsicles</td>
      <td>164.6</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>3</td>
      <td>0.548609</td>
    </tr>
    <tr>
      <th>5</th>
      <td>how i got my family to eat spinach&nbsp;&nbsp;spinach ca...</td>
      <td>166.1</td>
      <td>16.0</td>
      <td>19.0</td>
      <td>8</td>
      <td>0.605985</td>
    </tr>
    <tr>
      <th>6</th>
      <td>i stole the idea from mirj&nbsp;&nbsp;sesame noodles</td>
      <td>783.4</td>
      <td>46.0</td>
      <td>36.0</td>
      <td>8</td>
      <td>0.546714</td>
    </tr>
    <tr>
      <th>7</th>
      <td>immoral&nbsp;&nbsp;sandwich filling&nbsp;&nbsp;loose meat</td>
      <td>223.2</td>
      <td>22.0</td>
      <td>35.0</td>
      <td>8</td>
      <td>0.622181</td>
    </tr>
    <tr>
      <th>8</th>
      <td>jiffy&nbsp;&nbsp;extra moist carrot cake</td>
      <td>612.1</td>
      <td>49.0</td>
      <td>15.0</td>
      <td>11</td>
      <td>0.550995</td>
    </tr>
    <tr>
      <th>9</th>
      <td>land of nod&nbsp;&nbsp;cinnamon buns</td>
      <td>575.3</td>
      <td>18.0</td>
      <td>28.0</td>
      <td>6</td>
      <td>0.562096</td>
    </tr>
    <tr>
      <th>10</th>
      <td>love is in the air&nbsp;&nbsp;beef fondue&nbsp;&nbsp; sauces</td>
      <td>1615.2</td>
      <td>154.0</td>
      <td>182.0</td>
      <td>12</td>
      <td>0.637167</td>
    </tr>
    <tr>
      <th>11</th>
      <td>mennonite&nbsp;&nbsp;corn fritters</td>
      <td>67.1</td>
      <td>7.0</td>
      <td>3.0</td>
      <td>8</td>
      <td>0.596702</td>
    </tr>
    <tr>
      <th>12</th>
      <td>never weep&nbsp;&nbsp;whipped cream</td>
      <td>276.3</td>
      <td>45.0</td>
      <td>3.0</td>
      <td>4</td>
      <td>0.505088</td>
    </tr>
    <tr>
      <th>13</th>
      <td>now and later&nbsp;&nbsp;vegetarian empanadas</td>
      <td>477.1</td>
      <td>36.0</td>
      <td>26.0</td>
      <td>22</td>
      <td>0.665874</td>
    </tr>
    <tr>
      <th>14</th>
      <td>off the cob&nbsp;&nbsp;freezer corn</td>
      <td>517.2</td>
      <td>58.0</td>
      <td>17.0</td>
      <td>4</td>
      <td>0.493361</td>
    </tr>
  </tbody>
</table>
</div>



## 11. Health Score Distribution Analysis


```python
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Health Score Distribution', fontsize=16, fontweight='bold')

# Histogram
ax1 = axes[0]
health_scores = df['health_score'].dropna()
ax1.hist(health_scores, bins=60, color='#27ae60', alpha=0.8, edgecolor='white')
ax1.axvline(health_scores.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {health_scores.mean():.3f}')
ax1.axvline(health_scores.median(), color='blue', linestyle='--', linewidth=2, label=f'Median: {health_scores.median():.3f}')
ax1.set_title('Health Score Histogram', fontsize=13)
ax1.set_xlabel('Health Score')
ax1.set_ylabel('Number of Recipes')
ax1.legend(fontsize=11)

# Box plot
ax2 = axes[1]
bp = ax2.boxplot(health_scores, vert=True, patch_artist=True, widths=0.5)
bp['boxes'][0].set_facecolor('#27ae60')
bp['boxes'][0].set_alpha(0.7)
ax2.set_title('Health Score Box Plot', fontsize=13)
ax2.set_ylabel('Health Score')
ax2.set_xticklabels(['All Recipes'])

plt.tight_layout()
plt.show()
```


    
## 12. Top and Bottom Recipes by Health Score


```python
display_cols = ['name', 'calories', 'total_fat_pdv', 'protein_pdv', 
                'ingredient_diversity', 'health_score']

print("=" * 70)
print("TOP 15 HEALTHIEST RECIPES (Highest Health Score)")
print("=" * 70)
top_healthy = df.nlargest(15, 'health_score')[display_cols].reset_index(drop=True)
display(top_healthy)

print("\n" + "=" * 70)
print("BOTTOM 15 LEAST HEALTHY RECIPES (Lowest Health Score)")
print("=" * 70)
bottom_healthy = df.nsmallest(15, 'health_score')[display_cols].reset_index(drop=True)
display(bottom_healthy)
```

    ======================================================================
    TOP 15 HEALTHIEST RECIPES (Highest Health Score)
    ======================================================================
    


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
      <th>name</th>
      <th>calories</th>
      <th>total_fat_pdv</th>
      <th>protein_pdv</th>
      <th>ingredient_diversity</th>
      <th>health_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>cioppino&nbsp;&nbsp;seafood stew</td>
      <td>789.8</td>
      <td>39.0</td>
      <td>203.0</td>
      <td>19</td>
      <td>0.888066</td>
    </tr>
    <tr>
      <th>1</th>
      <td>spanish moroccan fish</td>
      <td>539.1</td>
      <td>18.0</td>
      <td>164.0</td>
      <td>15</td>
      <td>0.878170</td>
    </tr>
    <tr>
      <th>2</th>
      <td>beef of eye round roast</td>
      <td>532.4</td>
      <td>19.0</td>
      <td>160.0</td>
      <td>15</td>
      <td>0.871017</td>
    </tr>
    <tr>
      <th>3</th>
      <td>bahama mama chicken marinade</td>
      <td>488.8</td>
      <td>17.0</td>
      <td>158.0</td>
      <td>14</td>
      <td>0.864897</td>
    </tr>
    <tr>
      <th>4</th>
      <td>one pot pork roast dinner&nbsp;&nbsp;or beef</td>
      <td>709.0</td>
      <td>22.0</td>
      <td>179.0</td>
      <td>13</td>
      <td>0.859077</td>
    </tr>
    <tr>
      <th>5</th>
      <td>boiled shrimp</td>
      <td>579.5</td>
      <td>18.0</td>
      <td>207.0</td>
      <td>11</td>
      <td>0.857806</td>
    </tr>
    <tr>
      <th>6</th>
      <td>chicken lime soup</td>
      <td>883.8</td>
      <td>42.0</td>
      <td>245.0</td>
      <td>16</td>
      <td>0.849617</td>
    </tr>
    <tr>
      <th>7</th>
      <td>turkey brine and injection marinade</td>
      <td>936.5</td>
      <td>67.0</td>
      <td>224.0</td>
      <td>21</td>
      <td>0.847291</td>
    </tr>
    <tr>
      <th>8</th>
      <td>chicken cacciatore&nbsp;&nbsp;pressure cooker</td>
      <td>587.2</td>
      <td>31.0</td>
      <td>161.0</td>
      <td>14</td>
      <td>0.847217</td>
    </tr>
    <tr>
      <th>9</th>
      <td>good eats roast turkey</td>
      <td>733.7</td>
      <td>56.0</td>
      <td>185.0</td>
      <td>15</td>
      <td>0.842126</td>
    </tr>
    <tr>
      <th>10</th>
      <td>garlic and herb oven fried halibut</td>
      <td>635.4</td>
      <td>30.0</td>
      <td>182.0</td>
      <td>11</td>
      <td>0.841008</td>
    </tr>
    <tr>
      <th>11</th>
      <td>moroccan crock pot tajine</td>
      <td>553.1</td>
      <td>29.0</td>
      <td>149.0</td>
      <td>15</td>
      <td>0.840678</td>
    </tr>
    <tr>
      <th>12</th>
      <td>empress chicken</td>
      <td>752.1</td>
      <td>16.0</td>
      <td>168.0</td>
      <td>12</td>
      <td>0.839746</td>
    </tr>
    <tr>
      <th>13</th>
      <td>sunshine curry chicken</td>
      <td>628.8</td>
      <td>8.0</td>
      <td>172.0</td>
      <td>9</td>
      <td>0.839693</td>
    </tr>
    <tr>
      <th>14</th>
      <td>chicken piccata&nbsp;&nbsp;a delicious italian chicken dish</td>
      <td>742.6</td>
      <td>60.0</td>
      <td>154.0</td>
      <td>20</td>
      <td>0.836767</td>
    </tr>
  </tbody>
</table>
</div>


    
    ======================================================================
    BOTTOM 15 LEAST HEALTHY RECIPES (Lowest Health Score)
    ======================================================================
    


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
      <th>name</th>
      <th>calories</th>
      <th>total_fat_pdv</th>
      <th>protein_pdv</th>
      <th>ingredient_diversity</th>
      <th>health_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>clarified butter or&nbsp;&nbsp;ghee</td>
      <td>3255.2</td>
      <td>566.0</td>
      <td>7.0</td>
      <td>2</td>
      <td>0.012000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>veronica s lemon buttercream frosting</td>
      <td>3446.8</td>
      <td>354.0</td>
      <td>5.0</td>
      <td>4</td>
      <td>0.026218</td>
    </tr>
    <tr>
      <th>2</th>
      <td>paige s buttercream frosting</td>
      <td>3536.2</td>
      <td>285.0</td>
      <td>6.0</td>
      <td>4</td>
      <td>0.027933</td>
    </tr>
    <tr>
      <th>3</th>
      <td>homemade soft&nbsp;&nbsp;butter spread</td>
      <td>5182.3</td>
      <td>901.0</td>
      <td>7.0</td>
      <td>4</td>
      <td>0.029647</td>
    </tr>
    <tr>
      <th>4</th>
      <td>cake decorating icing</td>
      <td>3618.7</td>
      <td>301.0</td>
      <td>3.0</td>
      <td>5</td>
      <td>0.031613</td>
    </tr>
    <tr>
      <th>5</th>
      <td>buttercream frosting&nbsp;&nbsp;decorators icing</td>
      <td>4302.9</td>
      <td>252.0</td>
      <td>0.0</td>
      <td>5</td>
      <td>0.037834</td>
    </tr>
    <tr>
      <th>6</th>
      <td>papaya seed dressing</td>
      <td>3114.9</td>
      <td>443.0</td>
      <td>2.0</td>
      <td>7</td>
      <td>0.047546</td>
    </tr>
    <tr>
      <th>7</th>
      <td>chili oil&nbsp;&nbsp;hong you</td>
      <td>4501.6</td>
      <td>754.0</td>
      <td>28.0</td>
      <td>2</td>
      <td>0.048000</td>
    </tr>
    <tr>
      <th>8</th>
      <td>ming s perfectly simple coconut sorbet</td>
      <td>4537.4</td>
      <td>291.0</td>
      <td>28.0</td>
      <td>3</td>
      <td>0.056824</td>
    </tr>
    <tr>
      <th>9</th>
      <td>marshmallow buttercream frosting</td>
      <td>2786.3</td>
      <td>284.0</td>
      <td>7.0</td>
      <td>4</td>
      <td>0.060576</td>
    </tr>
    <tr>
      <th>10</th>
      <td>the redneck s best caramels</td>
      <td>3138.9</td>
      <td>274.0</td>
      <td>15.0</td>
      <td>6</td>
      <td>0.061008</td>
    </tr>
    <tr>
      <th>11</th>
      <td>homemade butter for the kitchen aid</td>
      <td>3284.4</td>
      <td>541.0</td>
      <td>39.0</td>
      <td>2</td>
      <td>0.066857</td>
    </tr>
    <tr>
      <th>12</th>
      <td>jeanie s cake frosting</td>
      <td>2718.6</td>
      <td>297.0</td>
      <td>5.0</td>
      <td>5</td>
      <td>0.072552</td>
    </tr>
    <tr>
      <th>13</th>
      <td>pecans on fire</td>
      <td>4762.4</td>
      <td>796.0</td>
      <td>21.0</td>
      <td>7</td>
      <td>0.080118</td>
    </tr>
    <tr>
      <th>14</th>
      <td>kittencal s method for clarified butter</td>
      <td>2170.1</td>
      <td>377.0</td>
      <td>5.0</td>
      <td>2</td>
      <td>0.099398</td>
    </tr>
  </tbody>
</table>
</div>


## 13. Health Score by Ingredient Count Bins


```python
# Create ingredient count bins
bins = [0, 3, 5, 8, 12, 20, 50]
labels = ['1-3', '4-5', '6-8', '9-12', '13-20', '21+']
df['ingredient_bin'] = pd.cut(df['ingredient_diversity'], bins=bins, labels=labels, right=True)

# Box plot of Health Score by ingredient bins
plt.figure(figsize=(12, 6))
ingredient_order = ['1-3', '4-5', '6-8', '9-12', '13-20', '21+']
sns.boxplot(data=df.dropna(subset=['health_score', 'ingredient_bin']), 
            x='ingredient_bin', y='health_score', order=ingredient_order,
            palette='viridis')
plt.title('Health Score Distribution by Number of Ingredients', fontsize=14, fontweight='bold')
plt.xlabel('Number of Ingredients', fontsize=12)
plt.ylabel('Health Score', fontsize=12)
plt.show()

# Summary statistics by bin
print("Mean Health Score by Ingredient Count:")
print(df.groupby('ingredient_bin', observed=False)['health_score'].agg(['mean', 'median', 'count']).to_string())
```


    
Mean Health Score by Ingredient Count:
                        mean    median  count
    ingredient_bin                           
    1-3             0.535305  0.548048   1878
    4-5             0.556358  0.563914   5908
    6-8             0.580654  0.585168  13314
    9-12            0.611857  0.614791  14003
    13-20           0.655462  0.661451   5643
    21+             0.701780  0.710671    222
    

## 14. Health Score vs Individual Nutritional Features


```python
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Health Score vs Normalized Nutritional Features', fontsize=16, fontweight='bold')

scatter_features = [
    ('protein_pdv_norm', 'Protein (norm)', '#3498db'),
    ('calories_norm', 'Calories (norm)', '#e74c3c'),
    ('total_fat_pdv_norm', 'Total Fat (norm)', '#f39c12'),
    ('ingredient_diversity_norm', 'Ingredient Diversity (norm)', '#2ecc71')
]

# Sample for scatter plot (to avoid overplotting)
sample_df = df.dropna(subset=['health_score']).sample(n=min(5000, len(df)), random_state=42)

for i, (col, label, color) in enumerate(scatter_features):
    ax = axes[i // 2][i % 2]
    ax.scatter(sample_df[col], sample_df['health_score'], 
              alpha=0.15, s=8, color=color)
    ax.set_xlabel(label, fontsize=11)
    ax.set_ylabel('Health Score', fontsize=11)
    ax.set_title(f'Health Score vs {label}', fontsize=12)
    
    # Add correlation annotation
    corr = df[col].corr(df['health_score'])
    ax.annotate(f'r = {corr:.3f}', xy=(0.05, 0.95), xycoords='axes fraction',
                fontsize=12, fontweight='bold', color='black',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.show()
```


    
## 15. Health Score Percentile Categorization


```python
# Pre-compute quantile thresholds ONCE (vectorized approach)
q80 = df['health_score'].quantile(0.80)
q60 = df['health_score'].quantile(0.60)
q40 = df['health_score'].quantile(0.40)
q20 = df['health_score'].quantile(0.20)

print(f'Quantile thresholds: Q80={q80:.4f}, Q60={q60:.4f}, Q40={q40:.4f}, Q20={q20:.4f}')

# Vectorized categorization using np.select (fast)
conditions = [
    df['health_score'].isna(),
    df['health_score'] >= q80,
    df['health_score'] >= q60,
    df['health_score'] >= q40,
    df['health_score'] >= q20,
]
choices = ['Unknown', 'Very Healthy', 'Healthy', 'Moderate', 'Less Healthy']
df['health_category'] = np.select(conditions, choices, default='Unhealthy')

# Distribution of categories
category_counts = df['health_category'].value_counts()
print('\nHealth Category Distribution:')
print(category_counts)

# Pie chart
plt.figure(figsize=(8, 8))
category_order = ['Very Healthy', 'Healthy', 'Moderate', 'Less Healthy', 'Unhealthy']
category_colors = ['#27ae60', '#2ecc71', '#f1c40f', '#e67e22', '#e74c3c']

counts = [category_counts.get(cat, 0) for cat in category_order]
plt.pie(counts, labels=category_order, colors=category_colors,
        autopct='%1.1f%%', startangle=90, textprops={'fontsize': 12})
plt.title('Distribution of Recipe Health Categories', fontsize=14, fontweight='bold')
plt.axis('equal')
plt.show()
```

    Quantile thresholds: Q80=0.6475, Q60=0.6094, Q40=0.5845, Q20=0.5588
    
    Health Category Distribution:
    health_category
    Moderate        8194
    Unhealthy       8194
    Very Healthy    8194
    Less Healthy    8193
    Healthy         8193
    Name: count, dtype: int64
    


    
## 16. Component Contribution Analysis

Visualize how each feature component contributes to the final Health Score.


```python
# Compute individual contributions
df['contribution_protein'] = W_PROTEIN * df['protein_pdv_norm']
df['contribution_calories'] = W_CALORIES * (1 - df['calories_norm'])
df['contribution_fat'] = W_FAT * (1 - df['total_fat_pdv_norm'])
df['contribution_diversity'] = W_DIVERSITY * df['ingredient_diversity_norm']

contribution_cols = ['contribution_protein', 'contribution_calories', 
                     'contribution_fat', 'contribution_diversity']

# Mean contributions
mean_contributions = df[contribution_cols].mean()
mean_contributions.index = ['Protein', 'Calories (inv.)', 'Fat (inv.)', 'Ing. Diversity']

plt.figure(figsize=(10, 6))
bars = plt.bar(mean_contributions.index, mean_contributions.values, 
               color=['#3498db', '#e74c3c', '#f39c12', '#2ecc71'],
               edgecolor='white', linewidth=1.5)

# Add value labels on bars
for bar, val in zip(bars, mean_contributions.values):
    plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.003,
             f'{val:.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.title('Average Component Contribution to Health Score', fontsize=14, fontweight='bold')
plt.ylabel('Mean Contribution', fontsize=12)
plt.xlabel('Feature Component', fontsize=12)
plt.ylim(0, max(mean_contributions.values) * 1.2)
plt.tight_layout()
plt.show()

print("\nMean Component Contributions:")
for name, val in mean_contributions.items():
    pct = (val / mean_contributions.sum()) * 100
    print(f"  {name:25s}: {val:.4f} ({pct:.1f}%)")
```


    
Mean Component Contributions:
      Protein                  : 0.0557 (9.3%)
      Calories (inv.)          : 0.2611 (43.8%)
      Fat (inv.)               : 0.2206 (37.0%)
      Ing. Diversity           : 0.0592 (9.9%)
    

## 17. Save Enriched Dataset


```python
# Select columns to save
save_columns = [
    'name', 'id', 'minutes', 'contributor_id', 'submitted', 'tags',
    'nutrition', 'n_steps', 'steps', 'description', 'ingredients', 'n_ingredients',
    'calories', 'total_fat_pdv', 'protein_pdv',
    'ingredient_diversity',
    'calories_norm', 'total_fat_pdv_norm', 'protein_pdv_norm', 'ingredient_diversity_norm',
    'health_score', 'health_category'
]

df_save = df[save_columns].copy()

print(f"Saving enriched dataset with Health Scores...")
print(f"Output path: {output_path}")
print(f"Shape: {df_save.shape}")
print(f"Columns: {df_save.columns.tolist()}")

df_save.to_csv(output_path, index=False)

print(f"\nSaved successfully!")
print(f"File size: {output_path.stat().st_size / (1024*1024):.2f} MB")
```

    Saving enriched dataset with Health Scores...
    Output path: ..\DATA\PROCESSED\food_metadata_with_health.csv
    Shape: (40968, 22)
    

    Columns: ['name', 'id', 'minutes', 'contributor_id', 'submitted', 'tags', 'nutrition', 'n_steps', 'steps', 'description', 'ingredients', 'n_ingredients', 'calories', 'total_fat_pdv', 'protein_pdv', 'ingredient_diversity', 'calories_norm', 'total_fat_pdv_norm', 'protein_pdv_norm', 'ingredient_diversity_norm', 'health_score', 'health_category']
    

    
    Saved successfully!
    File size: 54.16 MB
    

## 18. Summary

### Key Findings:

| Metric | Value |
|--------|-------|
| Total recipes processed | See output above |
| Features used | Normalized protein, calories, fat, ingredient diversity |
| Normalization | Min-Max with 99th percentile capping |
| Health Score range | [0, 1] |
| Health Score formula | 0.30 x protein + 0.30 x (1-calories) + 0.25 x (1-fat) + 0.15 x diversity |

### Design Decisions:
1. **Outlier capping** at 99th/1st percentile prevents extreme values from dominating normalization.
2. **Calories and fat are inverted** (1 - normalized value) because lower values indicate healthier recipes.
3. **Protein gets the highest weight** along with calories - these are the most impactful nutritional indicators.
4. **Ingredient diversity** gets a lower weight as it is a softer proxy for nutritional balance.
5. The Health Score output can be directly integrated into the recommendation pipeline as a re-ranking or filtering signal.

### Output:
- Enriched dataset saved to `dataset/archive_3/recipes_with_health_score.csv` with all original recipe fields plus:
  - Parsed nutritional features
  - Normalized features
  - Composite Health Score
  - Health Category (Very Healthy / Healthy / Moderate / Less Healthy / Unhealthy)
