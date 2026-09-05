# 01 - Dataset Loading

**Project:** GroundedNutriRec  
**Role:** Data + Preprocessing Lead  
**Scope:** Load and inspect the Food.com (RAW_recipes.csv, RAW_interactions.csv) and RecipeNLG datasets.  
**Goal:** Verify schemas, parse structured columns, and create 20K development samples.

## 1. Environment Setup


```python
import sys
from pathlib import Path

# Add project root to sys.path so that src modules are importable
project_root = Path.cwd().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np

from SRC.data_preprocessing import (
    load_raw_recipes,
    load_raw_interactions,
    load_recipenlg,
    create_development_sample,
    SAMPLE_DATA_DIR,
    RAW_DATA_DIR,
    NUTRITION_COLUMN_NAMES,
)

pd.set_option('display.max_columns', 30)
pd.set_option('display.max_colwidth', 80)
print(f'Project Root: {project_root}')
print(f'Raw Data Dir: {RAW_DATA_DIR}')
print(f'pandas {pd.__version__}, numpy {np.__version__}')
```

    Project Root: C:\Users\Kush Shah\OneDrive\Desktop\Internship
    Raw Data Dir: C:\Users\Kush Shah\OneDrive\Desktop\Internship\DATA\RAW
    pandas 3.0.0, numpy 2.4.1
    

## 2. Load RAW_recipes.csv

Food.com dataset with 231,637 recipes. Columns include stringified Python lists  
for `tags`, `ingredients`, `steps`, and `nutrition` (a 7-element PDV vector).  
The `load_raw_recipes` function parses these lists and decomposes the nutrition vector.


```python
# Load with full parsing: stringified lists -> Python lists, nutrition -> 7 columns
recipes_df = load_raw_recipes(parse_lists=True, parse_nutrition=True)
print(f'Shape: {recipes_df.shape}')
print(f'Columns: {list(recipes_df.columns)}')
```

    Shape: (231637, 19)
    Columns: ['name', 'id', 'minutes', 'contributor_id', 'submitted', 'tags', 'nutrition', 'n_steps', 'steps', 'description', 'ingredients', 'n_ingredients', 'calories', 'total_fat_pdv', 'sugar_pdv', 'sodium_pdv', 'protein_pdv', 'saturated_fat_pdv', 'carbohydrates_pdv']
    


```python
recipes_df.head(3)
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
      <th>ingredients</th>
      <th>n_ingredients</th>
      <th>calories</th>
      <th>total_fat_pdv</th>
      <th>sugar_pdv</th>
      <th>sodium_pdv</th>
      <th>protein_pdv</th>
      <th>saturated_fat_pdv</th>
      <th>carbohydrates_pdv</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>arriba&nbsp;&nbsp; baked winter squash mexican style</td>
      <td>137739</td>
      <td>55</td>
      <td>47892</td>
      <td>2005-09-16</td>
      <td>[60-minutes-or-less, time-to-make, course, main-ingredient, cuisine, prepara...</td>
      <td>[51.5, 0.0, 13.0, 0.0, 2.0, 0.0, 4.0]</td>
      <td>11</td>
      <td>[make a choice and proceed with recipe, depending on size of squash , cut in...</td>
      <td>autumn is my favorite time of year to cook! this recipe \r\ncan be prepared ...</td>
      <td>[winter squash, mexican seasoning, mixed spice, honey, butter, olive oil, salt]</td>
      <td>7</td>
      <td>51.5</td>
      <td>0.0</td>
      <td>13.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>a bit different&nbsp;&nbsp;breakfast pizza</td>
      <td>31490</td>
      <td>30</td>
      <td>26278</td>
      <td>2002-06-17</td>
      <td>[30-minutes-or-less, time-to-make, course, main-ingredient, cuisine, prepara...</td>
      <td>[173.4, 18.0, 0.0, 17.0, 22.0, 35.0, 1.0]</td>
      <td>9</td>
      <td>[preheat oven to 425 degrees f, press dough into the bottom and sides of a 1...</td>
      <td>this recipe calls for the crust to be prebaked a bit before adding ingredien...</td>
      <td>[prepared pizza crust, sausage patty, eggs, milk, salt and pepper, cheese]</td>
      <td>6</td>
      <td>173.4</td>
      <td>18.0</td>
      <td>0.0</td>
      <td>17.0</td>
      <td>22.0</td>
      <td>35.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>all in the kitchen&nbsp;&nbsp;chili</td>
      <td>112140</td>
      <td>130</td>
      <td>196586</td>
      <td>2005-02-25</td>
      <td>[time-to-make, course, preparation, main-dish, chili, crock-pot-slow-cooker,...</td>
      <td>[269.8, 22.0, 32.0, 48.0, 39.0, 27.0, 5.0]</td>
      <td>6</td>
      <td>[brown ground beef in large pot, add chopped onions to ground beef when almo...</td>
      <td>this modified version of 'mom's' chili was a hit at our 2004 christmas party...</td>
      <td>[ground beef, yellow onions, diced tomatoes, tomato paste, tomato soup, rote...</td>
      <td>13</td>
      <td>269.8</td>
      <td>22.0</td>
      <td>32.0</td>
      <td>48.0</td>
      <td>39.0</td>
      <td>27.0</td>
      <td>5.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
recipes_df.dtypes
```




    name                            str
    id                            int64
    minutes                       int64
    contributor_id                int64
    submitted            datetime64[us]
    tags                         object
    nutrition                    object
    n_steps                       int64
    steps                        object
    description                     str
    ingredients                  object
    n_ingredients                 int64
    calories                    float64
    total_fat_pdv               float64
    sugar_pdv                   float64
    sodium_pdv                  float64
    protein_pdv                 float64
    saturated_fat_pdv           float64
    carbohydrates_pdv           float64
    dtype: object




```python
# Verify that nutrition was correctly decomposed into 7 separate float columns
for col in NUTRITION_COLUMN_NAMES:
    assert col in recipes_df.columns, f'Missing nutrition column: {col}'
    assert recipes_df[col].dtype in ['float64', 'float32'], f'{col} is not numeric'

print('Nutrition decomposition verified successfully.')
recipes_df[NUTRITION_COLUMN_NAMES].describe()
```

    Nutrition decomposition verified successfully.
    




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
      <th>calories</th>
      <th>total_fat_pdv</th>
      <th>sugar_pdv</th>
      <th>sodium_pdv</th>
      <th>protein_pdv</th>
      <th>saturated_fat_pdv</th>
      <th>carbohydrates_pdv</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>231637.000000</td>
      <td>231637.00000</td>
      <td>231637.000000</td>
      <td>231637.000000</td>
      <td>231637.00000</td>
      <td>231637.000000</td>
      <td>231637.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>473.942425</td>
      <td>36.08070</td>
      <td>84.296865</td>
      <td>30.147485</td>
      <td>34.68186</td>
      <td>45.589150</td>
      <td>15.560403</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1189.711374</td>
      <td>77.79884</td>
      <td>800.080897</td>
      <td>131.961589</td>
      <td>58.47248</td>
      <td>98.235758</td>
      <td>81.824560</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>174.400000</td>
      <td>8.00000</td>
      <td>9.000000</td>
      <td>5.000000</td>
      <td>7.00000</td>
      <td>7.000000</td>
      <td>4.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>313.400000</td>
      <td>20.00000</td>
      <td>25.000000</td>
      <td>14.000000</td>
      <td>18.00000</td>
      <td>23.000000</td>
      <td>9.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>519.700000</td>
      <td>41.00000</td>
      <td>68.000000</td>
      <td>33.000000</td>
      <td>51.00000</td>
      <td>52.000000</td>
      <td>16.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>434360.200000</td>
      <td>17183.00000</td>
      <td>362729.000000</td>
      <td>29338.000000</td>
      <td>6552.00000</td>
      <td>10395.000000</td>
      <td>36098.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Check for null values across all columns
null_counts = recipes_df.isnull().sum()
print('Columns with null values:')
print(null_counts[null_counts > 0])
```

    Columns with null values:
    name              1
    description    4979
    dtype: int64
    


```python
# Verify parsed list columns - each should contain Python lists, not strings
for col in ['tags', 'ingredients', 'steps']:
    sample_value = recipes_df[col].dropna().iloc[0]
    assert isinstance(sample_value, list), f'{col} should be a list, got {type(sample_value)}'
    print(f'{col}: type={type(sample_value).__name__}, sample_length={len(sample_value)}')

print('\nAll list columns parsed correctly.')
```

    tags: type=list, sample_length=20
    ingredients: type=list, sample_length=7
    steps: type=list, sample_length=11
    
    All list columns parsed correctly.
    

## 3. Load RAW_interactions.csv

User-recipe interaction data: `user_id`, `recipe_id`, `date`, `rating`, `review`.  
Ratings range from 0 to 5 where 0 means no explicit rating was given.


```python
interactions_df = load_raw_interactions()
print(f'Shape: {interactions_df.shape}')
print(f'Columns: {list(interactions_df.columns)}')
```

    Shape: (1132367, 5)
    Columns: ['user_id', 'recipe_id', 'date', 'rating', 'review']
    


```python
interactions_df.head(3)
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
      <th>user_id</th>
      <th>recipe_id</th>
      <th>date</th>
      <th>rating</th>
      <th>review</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>38094</td>
      <td>40893</td>
      <td>2003-02-17</td>
      <td>4</td>
      <td>Great with a salad. Cooked on top of stove for 15 minutes.Added a shake of c...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1293707</td>
      <td>40893</td>
      <td>2011-12-21</td>
      <td>5</td>
      <td>So simple, so delicious! Great for chilly fall evening. Should have doubled ...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8937</td>
      <td>44394</td>
      <td>2002-12-01</td>
      <td>4</td>
      <td>This worked very well and is EASY.&nbsp;&nbsp;I used not quite a whole package (10oz) ...</td>
    </tr>
  </tbody>
</table>
</div>




```python
interactions_df.dtypes
```




    user_id               int64
    recipe_id             int64
    date         datetime64[us]
    rating                int64
    review                  str
    dtype: object




```python
# Rating distribution - critical for implicit feedback derivation
print('Rating value counts:')
print(interactions_df['rating'].value_counts().sort_index())
print(f'\nRating=0 (no explicit feedback): {(interactions_df["rating"] == 0).sum():,}')
```

    Rating value counts:
    rating
    0     60847
    1     12818
    2     14123
    3     40855
    4    187360
    5    816364
    Name: count, dtype: int64
    
    Rating=0 (no explicit feedback): 60,847
    


```python
# Unique user and recipe counts
unique_users = interactions_df['user_id'].nunique()
unique_recipes_interacted = interactions_df['recipe_id'].nunique()
total_interactions = len(interactions_df)
sparsity = 1 - (total_interactions / (unique_users * unique_recipes_interacted))

print(f'Unique users:    {unique_users:,}')
print(f'Unique recipes:  {unique_recipes_interacted:,}')
print(f'Total interactions: {total_interactions:,}')
print(f'Interaction matrix sparsity: {sparsity:.6f} ({sparsity*100:.4f}%)')
```

    Unique users:    226,570
    Unique recipes:  231,637
    Total interactions: 1,132,367
    Interaction matrix sparsity: 0.999978 (99.9978%)
    

## 4. Load RecipeNLG (Parquet)

RecipeNLG: 2.2M recipes scraped from recipe websites.  
Used as a supplementary corpus for ingredient vocabulary and NER training.  
Loading first rows only to inspect schema without loading 1 GB into memory.


```python
# Load first 5 rows to inspect schema without loading 1 GB into memory
recipenlg_sample = load_recipenlg(nrows=5)
print(f'Columns: {list(recipenlg_sample.columns)}')
recipenlg_sample.head()
```

    Columns: ['Unnamed: 0', 'title', 'ingredients', 'directions', 'link', 'source', 'NER']
    




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
      <th>Unnamed: 0</th>
      <th>title</th>
      <th>ingredients</th>
      <th>directions</th>
      <th>link</th>
      <th>source</th>
      <th>NER</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>No-Bake Nut Cookies</td>
      <td>["1 c. firmly packed brown sugar", "1/2 c. evaporated milk", "1/2 tsp. vanil...</td>
      <td>["In a heavy 2-quart saucepan, mix brown sugar, nuts, evaporated milk and bu...</td>
      <td>www.cookbooks.com/Recipe-Details.aspx?id=44874</td>
      <td>Gathered</td>
      <td>["brown sugar", "milk", "vanilla", "nuts", "butter", "bite size shredded ric...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>Jewell Ball'S Chicken</td>
      <td>["1 small jar chipped beef, cut up", "4 boned chicken breasts", "1 can cream...</td>
      <td>["Place chipped beef on bottom of baking dish.", "Place chicken on top of be...</td>
      <td>www.cookbooks.com/Recipe-Details.aspx?id=699419</td>
      <td>Gathered</td>
      <td>["beef", "chicken breasts", "cream of mushroom soup", "sour cream"]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>Creamy Corn</td>
      <td>["2 (16 oz.) pkg. frozen corn", "1 (8 oz.) pkg. cream cheese, cubed", "1/3 c...</td>
      <td>["In a slow cooker, combine all ingredients. Cover and cook on low for 4 hou...</td>
      <td>www.cookbooks.com/Recipe-Details.aspx?id=10570</td>
      <td>Gathered</td>
      <td>["frozen corn", "cream cheese", "butter", "garlic powder", "salt", "pepper"]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>Chicken Funny</td>
      <td>["1 large whole chicken", "2 (10 1/2 oz.) cans chicken gravy", "1 (10 1/2 oz...</td>
      <td>["Boil and debone chicken.", "Put bite size pieces in average size square ca...</td>
      <td>www.cookbooks.com/Recipe-Details.aspx?id=897570</td>
      <td>Gathered</td>
      <td>["chicken", "chicken gravy", "cream of mushroom soup", "shredded cheese"]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>Reeses Cups(Candy)</td>
      <td>["1 c. peanut butter", "3/4 c. graham cracker crumbs", "1 c. melted butter",...</td>
      <td>["Combine first four ingredients and press in 13 x 9-inch ungreased pan.", "...</td>
      <td>www.cookbooks.com/Recipe-Details.aspx?id=659239</td>
      <td>Gathered</td>
      <td>["peanut butter", "graham cracker crumbs", "butter", "powdered sugar", "choc...</td>
    </tr>
  </tbody>
</table>
</div>



## 5. Create Development Samples

20,000-row random samples saved to `data/sample/` for rapid  
iteration during EDA and model prototyping.


```python
SAMPLE_DATA_DIR.mkdir(parents=True, exist_ok=True)

recipes_sample = create_development_sample(
    recipes_df,
    sample_size=20000,
    random_seed=42,
    output_path=SAMPLE_DATA_DIR / 'recipes_sample.parquet',
)
print(f'Recipes sample shape: {recipes_sample.shape}')
print(f'Saved to: {SAMPLE_DATA_DIR / "recipes_sample.parquet"}')
```

    Recipes sample shape: (20000, 19)
    Saved to: C:\Users\Kush Shah\OneDrive\Desktop\Internship\DATA\SAMPLE\recipes_sample.parquet
    


```python
interactions_sample = create_development_sample(
    interactions_df,
    sample_size=20000,
    random_seed=42,
    output_path=SAMPLE_DATA_DIR / 'interactions_sample.parquet',
)
print(f'Interactions sample shape: {interactions_sample.shape}')
print(f'Saved to: {SAMPLE_DATA_DIR / "interactions_sample.parquet"}')
```

    Interactions sample shape: (20000, 5)
    Saved to: C:\Users\Kush Shah\OneDrive\Desktop\Internship\DATA\SAMPLE\interactions_sample.parquet
    

## 6. Summary

| Dataset | File | Rows | Columns | Key Features |
| --- | --- | --- | --- | --- |
| Food.com Recipes | RAW_recipes.csv | ~231K | 12 + 7 nutrition | Parsed lists, nutrition decomposed |
| Food.com Interactions | RAW_interactions.csv | ~1.1M | 5 | date parsed, rating distribution inspected |
| RecipeNLG | recipenlg.parquet | ~2.2M | varies | Supplementary corpus for NER |

**Next:** `02_EDA_FOOD_DATASET.ipynb` for detailed exploratory analysis.
