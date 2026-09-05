# 22. Multi-Objective Recommendation System Comparison
**Project:** GroundedNutriRec
**Task:** Evaluate the performance and trade-offs of all implemented ranking strategies.

This notebook plots and displays a comprehensive baseline comparison across all recommendation configurations.


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style='whitegrid')
print('Libraries imported.')
```

    Libraries imported.
    


```python
# Load metrics
metrics_data = {
    'Variant': [
        'Preference only',
        'Preference + Health',
        'Preference + Health + Time',
        'Full Multi-Objective',
        'Pareto-Front Ranking'
    ],
    'Precision@10': [0.000629, 0.000381, 0.000323, 0.003024, 0.006671],
    'Recall@10': [0.001767, 0.001108, 0.000953, 0.008576, 0.017495],
    'Avg Health Score': [0.6244, 0.7485, 0.7386, 0.7089, 0.6710],
    'Avg Prep Time (Mins)': [68.34, 90.11, 35.51, 27.98, 41.73],
    'Diversity (ILD@10)': [0.2942, 0.3341, 0.3352, 0.4027, 0.5883]
}

df_comparison = pd.DataFrame(metrics_data)
print('Comparison DataFrame defined:')
display(df_comparison)
```

    Comparison DataFrame defined:
    


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
      <th>Variant</th>
      <th>Precision@10</th>
      <th>Recall@10</th>
      <th>Avg Health Score</th>
      <th>Avg Prep Time (Mins)</th>
      <th>Diversity (ILD@10)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Preference only</td>
      <td>0.000629</td>
      <td>0.001767</td>
      <td>0.6244</td>
      <td>68.34</td>
      <td>0.2942</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Preference + Health</td>
      <td>0.000381</td>
      <td>0.001108</td>
      <td>0.7485</td>
      <td>90.11</td>
      <td>0.3341</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Preference + Health + Time</td>
      <td>0.000323</td>
      <td>0.000953</td>
      <td>0.7386</td>
      <td>35.51</td>
      <td>0.3352</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Full Multi-Objective</td>
      <td>0.003024</td>
      <td>0.008576</td>
      <td>0.7089</td>
      <td>27.98</td>
      <td>0.4027</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Pareto-Front Ranking</td>
      <td>0.006671</td>
      <td>0.017495</td>
      <td>0.6710</td>
      <td>41.73</td>
      <td>0.5883</td>
    </tr>
  </tbody>
</table>
</div>



```python
# Plot trade-off chart
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 1. Precision comparison
sns.barplot(data=df_comparison, x='Precision@10', y='Variant', ax=axes[0], palette='viridis')
axes[0].set_title('Precision@10 Comparison')
axes[0].set_xlabel('Precision@10')

# 2. Health score comparison
sns.barplot(data=df_comparison, x='Avg Health Score', y='Variant', ax=axes[1], palette='crest')
axes[1].set_title('Average Health Score Comparison')
axes[1].set_xlabel('Health Score')

# 3. Diversity comparison
sns.barplot(data=df_comparison, x='Diversity (ILD@10)', y='Variant', ax=axes[2], palette='mako')
axes[2].set_title('Diversity (ILD@10) Comparison')
axes[2].set_xlabel('ILD@10')

plt.tight_layout()
plt.savefig('../RESULTS/WEEK 05/multi_objective_comparison.png', dpi=300, bbox_inches='tight')
plt.show()
```

    C:\Users\Kush Shah\AppData\Local\Temp\ipykernel_12608\1709417865.py:5: FutureWarning: 
    
    Passing `palette` without assigning `hue` is deprecated and will be removed in v0.14.0. Assign the `y` variable to `hue` and set `legend=False` for the same effect.
    
      sns.barplot(data=df_comparison, x='Precision@10', y='Variant', ax=axes[0], palette='viridis')
    C:\Users\Kush Shah\AppData\Local\Temp\ipykernel_12608\1709417865.py:10: FutureWarning: 
    
    Passing `palette` without assigning `hue` is deprecated and will be removed in v0.14.0. Assign the `y` variable to `hue` and set `legend=False` for the same effect.
    
      sns.barplot(data=df_comparison, x='Avg Health Score', y='Variant', ax=axes[1], palette='crest')
    C:\Users\Kush Shah\AppData\Local\Temp\ipykernel_12608\1709417865.py:15: FutureWarning: 
    
    Passing `palette` without assigning `hue` is deprecated and will be removed in v0.14.0. Assign the `y` variable to `hue` and set `legend=False` for the same effect.
    
      sns.barplot(data=df_comparison, x='Diversity (ILD@10)', y='Variant', ax=axes[2], palette='mako')
    


    
