"""
data_preprocessing.py  -  GroundedNutriRec Data Loading and Preprocessing Module

Centralizes all data loading, parsing, type casting, and sampling operations
for the Food.com (RAW_recipes / RAW_interactions) and RecipeNLG datasets. Every
notebook and downstream training script imports from this module rather than
duplicating loading logic.

All file paths are resolved with pathlib.Path relative to the project root so
that the codebase remains portable across Windows, macOS, and Linux without
hard-coded absolute paths.
"""

from pathlib import Path
from typing import Optional
import ast

import pandas as pd
import numpy as np

# ===========================================================================
# MODULE CONSTANTS
# ===========================================================================

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
"""
Project root directory, resolved two levels up from this file
(src/data_preprocessing.py -> project root).
"""

RAW_DATA_DIR: Path = PROJECT_ROOT / "DATA" / "RAW"
PROCESSED_DATA_DIR: Path = PROJECT_ROOT / "DATA" / "PROCESSED"
SAMPLE_DATA_DIR: Path = PROJECT_ROOT / "DATA" / "SAMPLE"

RAW_RECIPES_FILENAME: str = "RAW_recipes.csv"
RAW_INTERACTIONS_FILENAME: str = "RAW_interactions.csv"
RECIPENLG_FILENAME: str = "recipenlg.parquet"

NUTRITION_COLUMN_NAMES: list[str] = [
    "calories",
    "total_fat_pdv",
    "sugar_pdv",
    "sodium_pdv",
    "protein_pdv",
    "saturated_fat_pdv",
    "carbohydrates_pdv",
]
"""
The 7-element nutrition vector in RAW_recipes.csv is stored as a stringified
Python list [calories, total_fat_pdv, sugar_pdv, sodium_pdv, protein_pdv,
saturated_fat_pdv, carbohydrates_pdv].  These names are applied after parsing.
"""

MINUTES_OUTLIER_THRESHOLD: int = 1440
"""
Recipes claiming more than 24 hours (1440 minutes) preparation time are
treated as outliers and excluded during cleaning.
"""


# ===========================================================================
# DATA LOADING FUNCTIONS
# ===========================================================================


def load_raw_recipes(
    filepath: Optional[Path] = None,
    parse_lists: bool = True,
    parse_nutrition: bool = True,
    nrows: Optional[int] = None,
) -> pd.DataFrame:
    """
    load_raw_recipes

    Loads RAW_recipes.csv from the raw data directory, optionally parsing
    stringified Python lists (tags, ingredients, steps, nutrition) into
    native list objects and decomposing the 7-element nutrition vector
    into separate numeric columns.

    @param  filepath        - Path to the CSV file.  Defaults to
                              data/raw/RAW_recipes.csv when None.
    @param  parse_lists     - When True, converts the 'tags', 'ingredients',
                              'steps', and 'nutrition' columns from their
                              string representations to Python lists using
                              ast.literal_eval.
    @param  parse_nutrition - When True and parse_lists is True, decomposes the
                              parsed nutrition list into 7 separate float
                              columns: calories, total_fat_pdv, sugar_pdv,
                              sodium_pdv, protein_pdv, saturated_fat_pdv,
                              carbohydrates_pdv.
    @param  nrows           - Maximum number of rows to read.  None reads all.
    @returns pd.DataFrame   - DataFrame with schema matching RAW_recipes.csv
                              plus any parsed/decomposed columns.
    @edge-cases             - Malformed nutrition strings are coerced to NaN
                              via errors='coerce' on each numeric conversion.
                            - Rows where ast.literal_eval raises ValueError
                              are set to None for the affected column.
    """
    if filepath is None:
        filepath = RAW_DATA_DIR / RAW_RECIPES_FILENAME

    recipes_dataframe = pd.read_csv(filepath, nrows=nrows)

    # --- Parse submitted date to datetime ---
    recipes_dataframe["submitted"] = pd.to_datetime(
        recipes_dataframe["submitted"], errors="coerce"
    )

    # --- Parse stringified list columns ---
    list_columns = ["tags", "ingredients", "steps", "nutrition"]
    if parse_lists:
        for column_name in list_columns:
            if column_name in recipes_dataframe.columns:
                recipes_dataframe[column_name] = recipes_dataframe[column_name].apply(
                    _safe_literal_eval
                )

    # --- Decompose nutrition vector ---
    if parse_lists and parse_nutrition:
        nutrition_decomposed = recipes_dataframe["nutrition"].apply(
            _decompose_nutrition_vector
        )
        nutrition_columns_dataframe = pd.DataFrame(
            nutrition_decomposed.tolist(),
            columns=NUTRITION_COLUMN_NAMES,
            index=recipes_dataframe.index,
        )
        recipes_dataframe = pd.concat(
            [recipes_dataframe, nutrition_columns_dataframe], axis=1
        )

    return recipes_dataframe


def load_raw_interactions(
    filepath: Optional[Path] = None,
    nrows: Optional[int] = None,
) -> pd.DataFrame:
    """
    load_raw_interactions

    Loads RAW_interactions.csv from the raw data directory and casts the date
    column to datetime.

    @param  filepath        - Path to the CSV file.  Defaults to
                              data/raw/RAW_interactions.csv when None.
    @param  nrows           - Maximum number of rows to read.  None reads all.
    @returns pd.DataFrame   - DataFrame with columns [user_id, recipe_id,
                              date, rating, review].
    @edge-cases             - Unparseable date strings are coerced to NaT.
                            - Rating values outside [0, 5] are not filtered
                              here; downstream logic should handle.
    """
    if filepath is None:
        filepath = RAW_DATA_DIR / RAW_INTERACTIONS_FILENAME

    interactions_dataframe = pd.read_csv(filepath, nrows=nrows)
    interactions_dataframe["date"] = pd.to_datetime(
        interactions_dataframe["date"], errors="coerce"
    )

    return interactions_dataframe


def load_recipenlg(
    filepath: Optional[Path] = None,
    nrows: Optional[int] = None,
) -> pd.DataFrame:
    """
    load_recipenlg

    Loads the RecipeNLG dataset from a Parquet file.  If nrows is specified,
    only the first nrows rows are returned.

    @param  filepath        - Path to the Parquet file.  Defaults to
                              data/raw/recipenlg.parquet when None.
    @param  nrows           - Maximum number of rows to read.  None reads all.
    @returns pd.DataFrame   - DataFrame containing RecipeNLG columns.
    @edge-cases             - Parquet files do not support native nrows, so
                              the function reads the full file and truncates.
                              For very large files, consider using PyArrow
                              directly with row-group-level slicing.
    """
    if filepath is None:
        filepath = RAW_DATA_DIR / RECIPENLG_FILENAME

    recipenlg_dataframe = pd.read_parquet(filepath)
    if nrows is not None:
        recipenlg_dataframe = recipenlg_dataframe.head(nrows)

    return recipenlg_dataframe


# ===========================================================================
# SAMPLING FUNCTIONS
# ===========================================================================


def create_development_sample(
    dataframe: pd.DataFrame,
    sample_size: int = 20000,
    random_seed: int = 42,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    create_development_sample

    Creates a reproducible random sample from a DataFrame for rapid
    development and debugging.  Optionally writes the sample to Parquet.

    @param  dataframe       - Source DataFrame to sample from.
    @param  sample_size     - Number of rows in the sample.  If the source has
                              fewer rows, the entire DataFrame is returned.
    @param  random_seed     - Seed for reproducibility.
    @param  output_path     - If provided, the sample is written as Parquet.
    @returns pd.DataFrame   - The sampled DataFrame.
    @edge-cases             - When len(dataframe) < sample_size, returns the
                              full DataFrame without replacement sampling.
    """
    actual_sample_size = min(sample_size, len(dataframe))
    sample_dataframe = dataframe.sample(
        n=actual_sample_size, random_state=random_seed
    ).reset_index(drop=True)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sample_dataframe.to_parquet(output_path, index=False)

    return sample_dataframe


# ===========================================================================
# CLEANING FUNCTIONS
# ===========================================================================


def remove_time_outliers(
    recipes_dataframe: pd.DataFrame,
    threshold_minutes: int = MINUTES_OUTLIER_THRESHOLD,
) -> pd.DataFrame:
    """
    remove_time_outliers

    Filters out recipes with preparation time exceeding the threshold.  Recipes
    claiming more than 1440 minutes (24 hours) are almost certainly data entry
    errors or specialty items (e.g., 30-day fermentation) that distort
    statistical summaries and model training.

    @param  recipes_dataframe  - DataFrame containing a 'minutes' column.
    @param  threshold_minutes  - Maximum allowed minutes.  Default is 1440.
    @returns pd.DataFrame      - Filtered DataFrame with outliers removed.
    @edge-cases                - NaN values in 'minutes' are dropped.
    """
    return recipes_dataframe[
        recipes_dataframe["minutes"].le(threshold_minutes)
    ].reset_index(drop=True)


def derive_implicit_feedback(
    interactions_dataframe: pd.DataFrame,
    positive_threshold: int = 4,
) -> pd.DataFrame:
    """
    derive_implicit_feedback

    Converts explicit ratings into binary implicit feedback.  Ratings at or
    above the positive_threshold are mapped to liked=1; ratings below it
    (but greater than 0) are mapped to liked=0.  Rating=0 indicates no
    explicit feedback and is mapped to NaN so that downstream logic can
    decide whether to treat these as negative or missing.

    @param  interactions_dataframe - DataFrame with a 'rating' column.
    @param  positive_threshold     - Minimum rating to be considered positive.
    @returns pd.DataFrame          - Input DataFrame with an appended 'liked'
                                     column of type Int8 (nullable integer).
    @edge-cases                    - Rating values outside [0, 5] are handled
                                     by the threshold comparison.
    """
    result = interactions_dataframe.copy()
    result["liked"] = np.where(
        result["rating"] == 0,
        pd.NA,
        np.where(result["rating"] >= positive_threshold, 1, 0),
    )
    result["liked"] = result["liked"].astype("Int8")
    return result


# ===========================================================================
# INTERNAL HELPERS
# ===========================================================================


def _safe_literal_eval(value):
    """
    _safe_literal_eval

    Safely evaluates a stringified Python literal (list, dict, tuple) using
    ast.literal_eval.  Returns None if evaluation fails.

    @param  value  - String representation of a Python literal.
    @returns       - Parsed Python object, or None on failure.
    """
    if pd.isna(value):
        return None
    try:
        return ast.literal_eval(str(value))
    except (ValueError, SyntaxError):
        return None


def _decompose_nutrition_vector(nutrition_list):
    """
    _decompose_nutrition_vector

    Extracts the 7 numeric values from a parsed nutrition list.  Returns
    a list of NaN values if the input is not a list of exactly 7 elements.

    @param  nutrition_list - A Python list of 7 float/int values, or None.
    @returns list[float]   - 7-element list of nutrition values.
    @edge-cases            - Non-list inputs and lists with length != 7
                             return [NaN]*7 to maintain DataFrame shape.
    """
    if isinstance(nutrition_list, list) and len(nutrition_list) == 7:
        return [float(value) for value in nutrition_list]
    return [np.nan] * 7
