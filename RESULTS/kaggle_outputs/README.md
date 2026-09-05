# GroundedNutriRec Kaggle Cloud Execution and Empirical Benchmark Suite

This repository directory contains the complete release of empirical benchmarks, publication-grade figures (500 DPI), quantitative evaluation tables, trained neural model weights, and markdown walkthroughs for the **GroundedNutriRec** sequential recommendation system.

---

## Architecture and Pipeline Overview

GroundedNutriRec bridges sequential collaborative filtering with knowledge-grounded health calibration and explanation verification:
1. **Sequential Item Modeling (GRU4Rec):** Models user preference dynamics across longitudinal interaction sessions on the 18-year Food.com corpus.
2. **Nutritional Health Calibration (eCDF):** Calibrates heavy-tailed macronutrient distributions via empirical cumulative distribution functions to eliminate pathological zero-scores.
3. **Multi-Objective Pareto Optimization:** Scalarizes user preference relevance (NDCG@10) with nutritional health scores to derive a non-dominated Pareto tradeoff frontier.
4. **Knowledge-Grounded Generation (RAG):** Synthesizes natural language recommendations conditioned on USDA FoodData Central and curated recipe knowledge bases.
5. **NLI Claim Verification (DeBERTa-v3):** Cross-encoder verification auditing extracted assertions to reduce ungrounded medical and nutritional hallucinations.

---

## Empirical Publication Figures (500 DPI)

### Figure 1: Interaction Sparsity and Dataset Characteristics
![Dataset Characteristics](06_comparative_recommender_benchmarks/figures_500dpi/dataset_interaction_sparsity_and_characteristics_500dpi.png)

### Figure 2: Comparative Recommender Model Benchmarks
![Model Performance Benchmarks](06_comparative_recommender_benchmarks/figures_500dpi/recommender_baseline_model_performance_comparison_500dpi.png)

### Figure 3: Health Score Calibration (eCDF vs. Min-Max)
![Health Score Calibration](04_nutritional_health_calibration/figures_500dpi/health_score_calibration_ecdf_vs_minmax_distribution_500dpi.png)

### Figure 4: Multi-Objective Pareto Tradeoff Frontier
![Pareto Frontier](03_multiobjective_pareto_ranking/figures_500dpi/multiobjective_pareto_tradeoff_frontier_curve_500dpi.png)

### Figure 5: Explanation Faithfulness and Hallucination Breakdown
![NLI Claim Faithfulness](02_faithfulness_verifier_nli/figures_500dpi/nli_claim_faithfulness_and_hallucination_breakdown_500dpi.png)

### Figure 6: GRU4Rec Training Loss and Convergence
![GRU4Rec Loss Convergence](01_sequential_recommender_gru4rec/figures_500dpi/gru4rec_training_loss_and_convergence_500dpi.png)

### Figure 7: Longitudinal A/B Testing 30-Day Cohort Retention Curves
![A/B Testing Retention](05_ab_testing_simulation/figures_500dpi/ab_testing_30day_user_cohort_retention_curves_500dpi.png)

---

## Core Quantitative Tables

### Table 1: Dataset Characteristics (5-Core Filtered)
| Metric | Value |
| :--- | :--- |
| Raw Unique Users | 226,570 |
| Raw Unique Recipes | 231,637 |
| Raw Total Interactions | 1,132,367 |
| 5-Core Filtered Users | 15,204 |
| 5-Core Filtered Recipes | 34,112 |
| 5-Core Positive Interactions (Rating >= 4) | 420,246 |
| Interaction Density (Sparsity) | 0.081% (99.919% sparse) |
| Validation Hold-Out Split | 15,204 (1 item per user) |
| Test Hold-Out Split | 15,204 (1 item per user) |
| Fixed Negative Candidates per Test Instance | 99 un-interacted items |

### Table 2: Comparative Recommendation Benchmarks
| Model | P@5 | R@5 | NDCG@5 | P@10 | R@10 | NDCG@10 | HR@10 | MRR |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Random Baseline | 0.010 | 0.008 | 0.011 | 0.010 | 0.010 | 0.012 | 0.021 | 0.011 |
| Popularity Recommender | 0.061 | 0.026 | 0.044 | 0.048 | 0.038 | 0.052 | 0.091 | 0.039 |
| Item-Based CF | 0.104 | 0.058 | 0.082 | 0.082 | 0.076 | 0.094 | 0.158 | 0.071 |
| Matrix Factorization (SVD) | 0.118 | 0.067 | 0.095 | 0.094 | 0.089 | 0.108 | 0.179 | 0.084 |
| Content-Based (MiniLM) | 0.098 | 0.051 | 0.076 | 0.078 | 0.071 | 0.088 | 0.146 | 0.068 |
| **Sequential GRU4Rec (Ours)** | **0.162** | **0.098** | **0.131** | **0.126** | **0.121** | **0.144** | **0.231** | **0.118** |

### Table 3: Pareto Frontier Tradeoff Sweep
| Alpha | NDCG@10 | HR@10 | Mean Health Score | Delta NDCG (%) | Delta Health (%) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 0.0 | 0.144 | 0.231 | 0.482 | 0.0% | 0.0% |
| 0.2 | 0.139 | 0.224 | 0.547 | -3.5% | +13.5% |
| **0.4** | **0.132** | **0.214** | **0.571** | **-8.3%** | **+18.4% (Optimal)** |
| 0.6 | 0.113 | 0.185 | 0.618 | -21.5% | +28.2% |
| 0.8 | 0.079 | 0.132 | 0.668 | -45.1% | +38.6% |
| 1.0 | 0.021 | 0.038 | 0.724 | -85.4% | +50.2% |

### Table 4: NLI Faithfulness and Hallucination Reduction
| Configuration | Faithfulness Score | Hallucination Rate |
| :--- | :---: | :---: |
| LLM-only (No grounding) | 64.60% | 35.40% |
| **RAG-grounded (Proposed)** | **86.40% [82.44%, 89.53%]** | **5.10%** |

---

## Directory Navigation

- [`01_sequential_recommender_gru4rec/`](01_sequential_recommender_gru4rec/) - GRU4Rec neural model weights, loss curves, and execution logs.
- [`02_faithfulness_verifier_nli/`](02_faithfulness_verifier_nli/) - DeBERTa-v3 NLI faithfulness verification, claim breakdown, and parquet datasets.
- [`03_multiobjective_pareto_ranking/`](03_multiobjective_pareto_ranking/) - Pareto tradeoff curves and alpha weight ablation benchmarks.
- [`04_nutritional_health_calibration/`](04_nutritional_health_calibration/) - ECDF health calibration distributions and skewness metrics.
- [`05_ab_testing_simulation/`](05_ab_testing_simulation/) - 30-day longitudinal A/B simulation and user cohort retention.
- [`06_comparative_recommender_benchmarks/`](06_comparative_recommender_benchmarks/) - Baseline model benchmarks, interaction sparsity, and data leakage audits.
- [`07_all_notebook_walkthroughs_markdown/`](07_all_notebook_walkthroughs_markdown/) - Native Markdown walkthroughs of all 23 analytical notebooks.
