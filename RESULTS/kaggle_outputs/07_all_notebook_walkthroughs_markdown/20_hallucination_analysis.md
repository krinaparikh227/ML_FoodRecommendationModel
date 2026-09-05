# 20 - Hallucination Analysis
**Project:** GroundedNutriRec
**Role:** LLM/RAG + Multi-Objective Ranking
**Task:** Aggregate verification results to calculate average Faithfulness Scores and Hallucination Rates, and generate evaluation reports and charts.



```python
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Define paths
INPUT_FILE = "data/interim/verified_claims.parquet"

RESULTS_DIR_WEEK5 = "../RESULTS/WEEK 05"
RESULTS_DIR_V5 = "../RESULTS/V5"
os.makedirs(RESULTS_DIR_WEEK5, exist_ok=True)
os.makedirs(RESULTS_DIR_V5, exist_ok=True)

```

## 1. Load Verified Claims
Load the verified claims dataset containing the support status for each claim.



```python
print("Loading verified claims dataset...")
df = pd.read_parquet(INPUT_FILE)
print(f"Loaded {len(df)} verified claims across recommendations.")

```

    Loading verified claims dataset...
    Loaded 407 verified claims across recommendations.
    

## 2. Aggregate Metrics per Recommendation Pair
Group the dataset by user_id and recipe_id to calculate the Faithfulness Score and Hallucination Rate for each recommended recipe.
*   `Faithfulness Score` = Supported Claims / Total Claims
*   `Hallucination Rate` = 1.0 - Faithfulness Score



```python
# Group and aggregate
pair_df = df.groupby(['explanation_id', 'recipe_id']).agg(
    total_claims=('verdict', 'count'),
    supported_claims=('verdict', lambda x: (x == 'Supported').sum())
).reset_index()

pair_df['faithfulness_score'] = pair_df['supported_claims'] / pair_df['total_claims']
pair_df['hallucination_rate'] = 1.0 - pair_df['faithfulness_score']

print("Aggregation per recommendation complete. Summary statistics:")
print(pair_df[['total_claims', 'supported_claims', 'faithfulness_score', 'hallucination_rate']].describe())

```

    Aggregation per recommendation complete. Summary statistics:
           total_claims  supported_claims  faithfulness_score  hallucination_rate
    count    100.000000        100.000000          100.000000          100.000000
    mean       4.070000          1.540000            0.382500            0.617500
    std        0.256432          0.999192            0.250492            0.250492
    min        4.000000          0.000000            0.000000            0.000000
    25%        4.000000          1.000000            0.250000            0.500000
    50%        4.000000          1.500000            0.325000            0.675000
    75%        4.000000          2.000000            0.500000            0.750000
    max        5.000000          4.000000            1.000000            1.000000
    

## 3. Calculate Global Evaluation Metrics
We compute the overall system-level metrics (averages across all evaluated recommendations).



```python
avg_faithfulness = pair_df['faithfulness_score'].mean()
avg_hallucination = pair_df['hallucination_rate'].mean()
avg_claims_per_rec = pair_df['total_claims'].mean()

print("\n=== GLOBAL EVALUATION REPORT ===")
print(f"Total Recommendations Evaluated: {len(pair_df)}")
print(f"Average Claims per Explanation:  {avg_claims_per_rec:.2f}")
print(f"System-Level Faithfulness Score: {avg_faithfulness:.4f}")
print(f"System-Level Hallucination Rate: {avg_hallucination:.4f}")

```

    
    === GLOBAL EVALUATION REPORT ===
    Total Recommendations Evaluated: 100
    Average Claims per Explanation:  4.07
    System-Level Faithfulness Score: 0.3825
    System-Level Hallucination Rate: 0.6175
    

## 4. Save Results and Plot Visualizations
Save the aggregated results table and generate charts showing faithfulness distribution and claim support percentages.



```python
# Save results tables
results_table_path_week5 = os.path.join(RESULTS_DIR_WEEK5, "hallucination_analysis_report.csv")
results_table_path_v5 = os.path.join(RESULTS_DIR_V5, "hallucination_analysis_report.csv")

pair_df.to_csv(results_table_path_week5, index=False)
pair_df.to_csv(results_table_path_v5, index=False)

# Compile a global summary dict to write as text report
summary_report_path = os.path.join(RESULTS_DIR_WEEK5, "summary_evaluation_report.txt")
with open(summary_report_path, "w", encoding="utf-8") as f:
    f.write("=== GroundedNutriRec RAG Evaluation Summary ===\n")
    f.write(f"Total Recommendations Evaluated: {len(pair_df)}\n")
    f.write(f"Average Claims per Explanation:  {avg_claims_per_rec:.2f}\n")
    f.write(f"Average Faithfulness Score:      {avg_faithfulness:.4f}\n")
    f.write(f"Average Hallucination Rate:      {avg_hallucination:.4f}\n")

print(f"Saved evaluation reports to {results_table_path_week5} and {summary_report_path}.")

```

    Saved evaluation reports to ../RESULTS/WEEK 05\hallucination_analysis_report.csv and ../RESULTS/WEEK 05\summary_evaluation_report.txt.
    


```python
# Plot 1: Distribution of Faithfulness Scores (Histogram)
plt.figure(figsize=(8, 5))
plt.hist(pair_df['faithfulness_score'], bins=10, color='forestgreen', alpha=0.8, edgecolor='black')
plt.title("Distribution of Faithfulness Scores across Recommendations")
plt.xlabel("Faithfulness Score")
plt.ylabel("Number of Recommendations")
plt.grid(axis='y', linestyle='--', alpha=0.7)

hist_path_week5 = os.path.join(RESULTS_DIR_WEEK5, "faithfulness_distribution.png")
hist_path_v5 = os.path.join(RESULTS_DIR_V5, "faithfulness_distribution.png")
plt.savefig(hist_path_week5, dpi=300, bbox_inches='tight')
plt.savefig(hist_path_v5, dpi=300, bbox_inches='tight')
plt.close()

# Plot 2: Supported vs Unsupported Claims (Bar Chart)
total_claims_count = df['verdict'].count()
supported_count = (df['verdict'] == 'Supported').sum()
unsupported_count = (df['verdict'] == 'Unsupported').sum()

plt.figure(figsize=(6, 5))
labels = ['Supported Claims', 'Unsupported Claims']
counts = [supported_count, unsupported_count]
colors = ['teal', 'indianred']

plt.bar(labels, counts, color=colors, alpha=0.8, edgecolor='black', width=0.5)
plt.title("Factual Support breakdown for Explanation Claims")
plt.ylabel("Number of Claims")
for i, v in enumerate(counts):
    plt.text(i, v + (total_claims_count * 0.01), f"{v} ({v/total_claims_count*100:.1f}%)", ha='center', fontweight='bold')

bar_path_week5 = os.path.join(RESULTS_DIR_WEEK5, "claim_support_breakdown.png")
bar_path_v5 = os.path.join(RESULTS_DIR_V5, "claim_support_breakdown.png")
plt.savefig(bar_path_week5, dpi=300, bbox_inches='tight')
plt.savefig(bar_path_v5, dpi=300, bbox_inches='tight')
plt.close()

print(f"Saved visualization charts to {hist_path_week5} and {bar_path_week5}.")

```

    Saved visualization charts to ../RESULTS/WEEK 05\faithfulness_distribution.png and ../RESULTS/WEEK 05\claim_support_breakdown.png.
    
