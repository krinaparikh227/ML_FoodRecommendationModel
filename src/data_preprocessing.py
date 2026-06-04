import os
import pandas as pd
import numpy as np

def load_food_dataset(raw_dir):
    """
    load_food_dataset
    
    Loads the Food.com recipes and reviews datasets. Checks for Parquet files first
    for optimized loading, falling back to CSV files if Parquet is not found.
    
    @param  {str} raw_dir  - Path to the raw data directory containing the datasets.
    @returns {tuple}       - A tuple containing (df_recipes, df_reviews) as pandas DataFrames.
    @validates             - Verifies the existence of files before attempting to load them.
    @edge-cases            - If files are missing, prints an error message and returns (None, None).
    """
    recipes_csv = os.path.join(raw_dir, 'recipes.csv')
    recipes_parquet = os.path.join(raw_dir, 'recipes.parquet')
    reviews_csv = os.path.join(raw_dir, 'reviews.csv')
    reviews_parquet = os.path.join(raw_dir, 'reviews.parquet')
    
    df_recipes = None
    df_reviews = None
    
    # Load Recipes
    if os.path.exists(recipes_parquet):
        print("Loading recipes from Parquet...")
        df_recipes = pd.read_parquet(recipes_parquet)
    elif os.path.exists(recipes_csv):
        print("Loading recipes from CSV...")
        df_recipes = pd.read_csv(recipes_csv)
    else:
        print(f"WARNING: Recipes dataset not found at {recipes_csv} or {recipes_parquet}")
        
    # Load Reviews
    if os.path.exists(reviews_parquet):
        print("Loading reviews from Parquet...")
        df_reviews = pd.read_parquet(reviews_parquet)
    elif os.path.exists(reviews_csv):
        print("Loading reviews from CSV...")
        df_reviews = pd.read_csv(reviews_csv)
    else:
        print(f"WARNING: Reviews dataset not found at {reviews_csv} or {reviews_parquet}")
        
    return df_recipes, df_reviews

def load_recipenlg(raw_dir):
    """
    load_recipenlg
    
    Scans the raw data directory for the RecipeNLG dataset CSV or Parquet files.
    If found, loads the dataset. If not found, prints download instructions.
    
    @param  {str} raw_dir  - Path to the raw data directory.
    @returns {DataFrame}   - The loaded RecipeNLG DataFrame or None if not found.
    @validates             - Checks for candidate files containing 'nlg' or 'recipenlg'.
    """
    nlg_candidates = [
        f for f in os.listdir(raw_dir)
        if ('nlg' in f.lower() or 'recipenlg' in f.lower() or f.lower() == 'dataset.csv')
        and not f.endswith('.zip')
    ]
    
    if not nlg_candidates:
        print("WARNING: RecipeNLG dataset not found in raw directory.")
        print("→ Download from: https://www.kaggle.com/datasets/paultimothymooney/recipenlg")
        print("→ Place extracted 'dataset.csv' into raw data folder as 'recipenlg.csv'.")
        return None
        
    nlg_file = os.path.join(raw_dir, nlg_candidates[0])
    if nlg_file.endswith('.parquet'):
        print(f"Loading RecipeNLG from Parquet: {nlg_file}")
        return pd.read_parquet(nlg_file)
    else:
        print(f"Loading RecipeNLG from CSV: {nlg_file}")
        return pd.read_csv(nlg_file)

def create_sampled_data(df_recipes, df_reviews, sample_dir, n_recipes=20000):
    """
    create_sampled_data
    
    Generates user-wise sampled subsets of the recipes and reviews for fast prototyping.
    Saves the generated samples in Parquet format within the sample directory.
    
    @param  {DataFrame} df_recipes - Raw recipes DataFrame.
    @param  {DataFrame} df_reviews - Raw reviews DataFrame.
    @param  {str} sample_dir       - Directory path to save the samples.
    @param  {int} n_recipes        - Number of recipes to sample.
    @returns {tuple}               - Tuple of (df_recipes_sample, df_reviews_sample)
    """
    os.makedirs(sample_dir, exist_ok=True)
    sample_recipes_path = os.path.join(sample_dir, 'recipes_sample.parquet')
    sample_reviews_path = os.path.join(sample_dir, 'reviews_sample.parquet')
    
    # Sample recipes
    df_recipes_sample = df_recipes.sample(n=n_recipes, random_state=42)
    sample_recipe_ids = set(df_recipes_sample['RecipeId'])
    
    # Filter reviews matching the sampled recipes
    df_reviews_sample = df_reviews[df_reviews['RecipeId'].isin(sample_recipe_ids)]
    
    # Save samples
    df_recipes_sample.to_parquet(sample_recipes_path, index=False)
    df_reviews_sample.to_parquet(sample_reviews_path, index=False)
    print(f"Successfully generated and saved samples at {sample_dir}")
    
    return df_recipes_sample, df_reviews_sample
