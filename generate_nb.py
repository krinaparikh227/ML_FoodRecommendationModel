import json

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Basic Exploratory Data Analysis (EDA)\n",
                "This notebook performs basic EDA on the recipes and reviews dataset from the `archive` folder, as well as the `full_dataset.csv` from the `dataset` folder, to answer the following questions:\n",
                "1. Number of users\n",
                "2. Number of recipes\n",
                "3. Number of interactions\n",
                "4. Rating distribution\n",
                "5. Top recipes\n",
                "6. Missing values\n",
                "7. Recipe categories\n",
                "8. Nutrition field availability"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Part 1: Archive Dataset Analysis\n",
                "These datasets (`recipes.csv` and `reviews.csv`) contain the specific fields required for user interaction, ratings, and nutrition analysis."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import pandas as pd\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "\n",
                "# Load archive data\n",
                "recipes = pd.read_csv('archive/recipes.csv')\n",
                "reviews = pd.read_csv('archive/reviews.csv')"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 1. Number of users\n",
                "recipe_users = set(recipes['AuthorId'].dropna().unique())\n",
                "review_users = set(reviews['AuthorId'].dropna().unique())\n",
                "total_users = len(recipe_users.union(review_users))\n",
                "print(f\"Total unique users: {total_users}\")"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 2. Number of recipes\n",
                "total_recipes = recipes['RecipeId'].nunique()\n",
                "print(f\"Total recipes: {total_recipes}\")"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 3. Number of interactions\n",
                "total_interactions = len(reviews)\n",
                "print(f\"Total interactions (reviews): {total_interactions}\")"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 4. Rating distribution\n",
                "plt.figure(figsize=(8, 5))\n",
                "sns.countplot(data=reviews, x='Rating', palette='viridis')\n",
                "plt.title('Rating Distribution (Reviews)')\n",
                "plt.xlabel('Rating')\n",
                "plt.ylabel('Count')\n",
                "plt.show()\n",
                "\n",
                "plt.figure(figsize=(8, 5))\n",
                "sns.histplot(recipes['AggregatedRating'].dropna(), bins=10, kde=False)\n",
                "plt.title('Aggregated Rating Distribution (Recipes)')\n",
                "plt.xlabel('Rating')\n",
                "plt.ylabel('Count')\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 5. Top recipes\n",
                "# Based on review count and aggregated rating\n",
                "top_recipes = recipes.sort_values(by=['ReviewCount', 'AggregatedRating'], ascending=[False, False]).head(10)\n",
                "print(\"Top 10 Recipes:\")\n",
                "display(top_recipes[['Name', 'ReviewCount', 'AggregatedRating']])"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 6. Missing values\n",
                "print(\"Missing values in Recipes:\")\n",
                "print(recipes.isnull().sum())\n",
                "print(\"\\nMissing values in Reviews:\")\n",
                "print(reviews.isnull().sum())"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 7. Recipe categories\n",
                "print(\"Top 20 Recipe Categories:\")\n",
                "print(recipes['RecipeCategory'].value_counts().head(20))\n",
                "\n",
                "plt.figure(figsize=(12, 6))\n",
                "recipes['RecipeCategory'].value_counts().head(20).plot(kind='bar', color='skyblue')\n",
                "plt.title('Top 20 Recipe Categories')\n",
                "plt.xlabel('Category')\n",
                "plt.ylabel('Count')\n",
                "plt.xticks(rotation=45, ha='right')\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 8. Nutrition field availability\n",
                "nutrition_cols = ['Calories', 'FatContent', 'SaturatedFatContent', 'CholesterolContent', \n",
                "                  'SodiumContent', 'CarbohydrateContent', 'FiberContent', 'SugarContent', 'ProteinContent']\n",
                "\n",
                "nutrition_missing = recipes[nutrition_cols].isnull().sum()\n",
                "nutrition_availability = len(recipes) - nutrition_missing\n",
                "\n",
                "print(\"Nutrition Field Availability (Number of Non-Null Entries):\")\n",
                "print(nutrition_availability)\n",
                "\n",
                "print(\"\\nNutrition Field Availability (%):\")\n",
                "print((nutrition_availability / len(recipes)) * 100)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Part 2: Full Dataset Analysis (`dataset/full_dataset.csv`)\n",
                "We will go through all 8 EDA tasks for the full dataset to ensure nothing is skipped. Note that some fields are not available in this schema."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Load full dataset\n",
                "full_dataset = pd.read_csv('dataset/full_dataset.csv')\n",
                "print(f\"Total rows in full_dataset: {len(full_dataset)}\")\n",
                "full_dataset.head()"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 1. Number of users\n",
                "print(\"Task 1: Number of users - Not available in full_dataset.csv as there is no AuthorId or user information.\")"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 2. Number of recipes\n",
                "if 'title' in full_dataset.columns:\n",
                "    total_full_recipes = full_dataset['title'].nunique()\n",
                "    print(f\"Task 2: Total unique recipe titles in full_dataset: {total_full_recipes}\")\n",
                "else:\n",
                "    print(f\"Task 2: Total recipes in full_dataset (by row count): {len(full_dataset)}\")"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 3. Number of interactions\n",
                "print(\"Task 3: Number of interactions - Not available in full_dataset.csv as there is no review or interaction data.\")"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 4. Rating distribution\n",
                "print(\"Task 4: Rating distribution - Not available in full_dataset.csv as there are no rating columns.\")"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 5. Top recipes\n",
                "print(\"Task 5: Top recipes - Cannot determine by rating. Displaying first 5 recipes instead:\")\n",
                "display(full_dataset.head(5))"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 6. Missing values\n",
                "print(\"Task 6: Missing values in full_dataset:\")\n",
                "print(full_dataset.isnull().sum())"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 7. Recipe categories\n",
                "print(\"Task 7: Recipe categories - Using 'source' as a proxy for category.\")\n",
                "if 'source' in full_dataset.columns:\n",
                "    print(\"\\nTop Sources:\")\n",
                "    print(full_dataset['source'].value_counts().head(20))\n",
                "    \n",
                "    plt.figure(figsize=(12, 6))\n",
                "    full_dataset['source'].value_counts().head(20).plot(kind='bar', color='lightgreen')\n",
                "    plt.title('Top 20 Sources in full_dataset (Proxy for Category)')\n",
                "    plt.xlabel('Source')\n",
                "    plt.ylabel('Count')\n",
                "    plt.xticks(rotation=45, ha='right')\n",
                "    plt.show()\n",
                "else:\n",
                "    print(\"Source column not available.\")"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 8. Nutrition field availability\n",
                "print(\"Task 8: Nutrition field availability - Not available in full_dataset.csv as there are no nutrition columns.\")"
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.8.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open('d:/Internship/Basic_EDA.ipynb', 'w') as f:
    json.dump(notebook, f, indent=2)

print("Notebook Basic_EDA.ipynb updated successfully to include dataset folder.")
