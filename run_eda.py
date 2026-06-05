import pandas as pd
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
import seaborn as sns
import os

output_md = 'EDA_results.md'

with open(output_md, 'w') as f:
    f.write('# Basic EDA Results\n\n')

def log(text):
    print(text)
    with open(output_md, 'a') as f:
        f.write(text + '\n')

log("Loading archive datasets...")
recipes = pd.read_csv('archive/recipes.csv')
reviews = pd.read_csv('archive/reviews.csv')
log("Loaded recipes and reviews.")

# 1. Number of users
recipe_users = set(recipes['AuthorId'].dropna().unique())
review_users = set(reviews['AuthorId'].dropna().unique())
total_users = len(recipe_users.union(review_users))
log(f"\n## 1. Number of Users\nTotal unique users (authors and reviewers): **{total_users}**\n")

# 2. Number of recipes
total_recipes = recipes['RecipeId'].nunique()
log(f"## 2. Number of Recipes\nTotal recipes: **{total_recipes}**\n")

# 3. Number of interactions
total_interactions = len(reviews)
log(f"## 3. Number of Interactions\nTotal interactions (reviews): **{total_interactions}**\n")

# 4. Rating distribution
log("## 4. Rating Distribution\nGenerating plots `rating_distribution_reviews.png` and `rating_distribution_recipes.png`...")
plt.figure(figsize=(8, 5))
sns.countplot(data=reviews, x='Rating', palette='viridis')
plt.title('Rating Distribution (Reviews)')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.savefig('rating_distribution_reviews.png')
plt.close()

plt.figure(figsize=(8, 5))
sns.histplot(recipes['AggregatedRating'].dropna(), bins=10, kde=False)
plt.title('Aggregated Rating Distribution (Recipes)')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.savefig('rating_distribution_recipes.png')
plt.close()
log("Plots saved.\n")

# 5. Top recipes
top_recipes = recipes.sort_values(by=['ReviewCount', 'AggregatedRating'], ascending=[False, False]).head(10)
log("## 5. Top Recipes")
log("Based on ReviewCount and AggregatedRating:")
log("```\n" + top_recipes[['Name', 'ReviewCount', 'AggregatedRating']].to_string() + "\n```\n")

# 6. Missing values
log("## 6. Missing Values")
log("### In Recipes:")
log("```\n" + recipes.isnull().sum().to_string() + "\n```\n")
log("### In Reviews:")
log("```\n" + reviews.isnull().sum().to_string() + "\n```\n")

# 7. Recipe categories
log("## 7. Recipe Categories")
log("Top 20 Recipe Categories:")
log("```\n" + recipes['RecipeCategory'].value_counts().head(20).to_string() + "\n```\n")

plt.figure(figsize=(12, 6))
recipes['RecipeCategory'].value_counts().head(20).plot(kind='bar', color='skyblue')
plt.title('Top 20 Recipe Categories')
plt.xlabel('Category')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('top_recipe_categories.png')
plt.close()
log("Category plot saved as `top_recipe_categories.png`.\n")

# 8. Nutrition field availability
nutrition_cols = ['Calories', 'FatContent', 'SaturatedFatContent', 'CholesterolContent', 
                  'SodiumContent', 'CarbohydrateContent', 'FiberContent', 'SugarContent', 'ProteinContent']
nutrition_missing = recipes[nutrition_cols].isnull().sum()
nutrition_availability = len(recipes) - nutrition_missing
log("## 8. Nutrition Field Availability")
log("Number of Non-Null Entries:")
log("```\n" + nutrition_availability.to_string() + "\n```\n")
log("Availability Percentage (%):")
log("```\n" + ((nutrition_availability / len(recipes)) * 100).to_string() + "\n```\n")

log("## Part 2: Full Dataset (dataset/full_dataset.csv)")
log("Loading full dataset...")
# Using chunking or nrows maybe if it's too large, but 2GB should load if memory > 8GB
try:
    full_dataset = pd.read_csv('dataset/full_dataset.csv')
    log(f"Total rows in full_dataset: **{len(full_dataset)}**\n")
    log("### Missing Values in full_dataset:")
    log("```\n" + full_dataset.isnull().sum().to_string() + "\n```\n")
    
    if 'source' in full_dataset.columns:
        log("### Top Sources:")
        log("```\n" + full_dataset['source'].value_counts().head(10).to_string() + "\n```\n")
        plt.figure(figsize=(10, 5))
        full_dataset['source'].value_counts().head(10).plot(kind='bar', color='lightgreen')
        plt.title('Top 10 Sources in full_dataset')
        plt.xlabel('Source')
        plt.ylabel('Count')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('top_sources_full_dataset.png')
        plt.close()
        log("Source plot saved as `top_sources_full_dataset.png`.\n")
except Exception as e:
    log(f"Error loading full_dataset.csv: {str(e)}")

print("Done. Results written to EDA_results.md")
