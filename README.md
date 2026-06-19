# GroundedNutriRec: Retrieval-Augmented Multi-Objective LLM Framework for Health-Aware and Explainable Food Recommendation

GroundedNutriRec is an advanced, research-oriented food recommendation framework that combines collaborative filtering models, multi-objective optimization (preference, healthiness, popularity, preparation time, and diversity), and Retrieval-Augmented Generation (RAG) to recommend recipes along with verified, hallucination-free explanations.

---

## 1. Project Architecture Overview

The system is organized into decoupled layers following standard Software Engineering patterns:

- **Data Layer:** Handles raw CSV and Parquet formats, applying memory-efficient chunked parsing and reproducible sampling.
- **Recommendation Engine:** Evaluates collaborative, content-based, and sequential baselines (including popularity-based filters and matrix factorization).
- **Ranking Optimization:** Balances taste preferences with nutritional guidelines (WHO/FSA traffic light ratings) and cooking complexity.
- **Explainability (RAG):** Anchors Large Language Model responses in recipe database evidence using vector embeddings (FAISS) to explain recommended meals.
- **Verification Layer:** Runs claims extraction and Natural Language Inference (NLI) to calculate explanation faithfulness, preventing hallucinated diet recommendations.

---

## 2. Directory Layout

The repository uses a fully capitalized naming structure for directory organization:

```
GroundedNutriRec/
│
├── APP/                    # Streamlit web application files
├── DATA/                   # Local-only dataset directories (not tracked by Git)
│   ├── RAW/                # Raw datasets (Food.com, RecipeNLG)
│   ├── PROCESSED/          # Preprocessed data and train-test splits
│   └── SAMPLE/             # 20k development samples for prototyping
├── NOTEBOOKS/              # Jupyter notebooks for weekly tasks
│   ├── 01_DATASET_LOADING.ipynb       # Week 1: Schema validation and sampling
│   ├── 02_FOOD_DATASET_EDA.ipynb      # Week 1: Exploratory data analysis
│   ├── 03_DATA_PREPROCESSING.ipynb    # Week 2: K-core filtering and split generation
│   └── 04_FEATURE_ENGINEERING.ipynb   # Week 2: WHO health score and categorical features
├── REPORTS_AND_DOCS/       # Academic literature notes and report documents
│   ├── Week1_Report.pdf    # Week 1 technical report (PDF)
│   ├── Week1_Report.docx   # Week 1 technical report (Word)
│   ├── Week1_Report.md     # Week 1 technical report (Markdown)
│   ├── Week01_Notes.pdf    # Week 1 lecture notes
│   ├── Week2_Report.pdf    # Week 2 technical report (PDF)
│   └── Week2_Report.md     # Week 2 technical report (Markdown)
├── RESULTS/                # Output artifacts and tables
│   ├── FIGURES/            # Distribution plots and charts (Week 1 EDA + Week 2 features)
│   ├── TABLES/             # Baseline performance evaluation metrics
│   └── WEEK 1/             # Screenshots from Week 1 EDA outputs
├── SRC/                    # Source code package
│   ├── EVALUATION/         # Metrics (Precision@K, NDCG@K, MRR@K)
│   ├── RAG/                # FAISS indexing and explanation prompt logic
│   ├── RECOMMENDERS/       # Collaborative filtering and sequential baselines
│   ├── __init__.py         # Package initialization
│   ├── data_preprocessing.py # Shared preprocessing: k-core, splits, feature engineering
│   └── utils.py            # Logger and path configuration helpers
├── requirements.txt        # Python dependencies
└── README.md               # Repository documentation
```

---

## 3. Weekly Research Roadmap

The project is structured into a 6-week summer program to progress from dataset understanding to integrated prototype verification:

| Week | Goal | Status | Deliverables |
| --- | --- | --- | --- |
| **Week 1** | Foundation and Dataset Exploration | **Complete** | Literature notes, schema validations, EDA plots, and sampling (`01_DATASET_LOADING.ipynb`, `02_FOOD_DATASET_EDA.ipynb`) |
| **Week 2** | Preprocessing and Feature Engineering | **Complete** | K-core filtering, implicit labels, temporal splits, WHO health score, categorical levels (`03_DATA_PREPROCESSING.ipynb`, `04_FEATURE_ENGINEERING.ipynb`) |
| **Week 3** | Baselines and Evaluation Metrics | Planned | Matrix factorization, similarity recommenders, and ranking metrics (`05_popularity_recommender.ipynb`, `06_content_based_recommender.ipynb`, `07_collaborative_filtering.ipynb`, `08_evaluation_metrics.ipynb`) |
| **Week 4** | Health-Aware and Multi-Objective Ranking | Planned | Food health scores, multi-objective ranking functions, and ablation weights (`09_health_score_generation.ipynb`, `10_multi_objective_ranking.ipynb`, `11_ablation_weight_analysis.ipynb`, `21_pareto_ranking.ipynb`) |
| **Week 5** | LLM/RAG and Faithfulness Check | Planned | Vector index pipelines, explanation generation, NLI claim verification (`12_food_knowledge_base_creation.ipynb`, `13_vector_search_rag.ipynb`, `14_llm_explanation_generation.ipynb`, `18_claim_extraction.ipynb`, `19_faithfulness_verification.ipynb`, `20_hallucination_analysis.ipynb`) |
| **Week 6** | System Integration and Demos | Planned | Sequential model implementations, offline A/B testing simulator, Streamlit interface, and final report (`app/streamlit_app.py`, `16_sequential_recommender_gru.ipynb`, `17_transformer_based_recommender.ipynb`, `23_ab_testing_simulator.ipynb`) |

---

## 4. Current Progress

### Week 1 — Foundation and Dataset Exploration (Complete)

- Validated schema integrity across the Food.com Recipes and Reviews datasets and the RecipeNLG corpus.
- Identified and documented dataset structure: 231,637 unique recipes, 1,132,367 interaction records, 226,570 unique users.
- Generated reproducible 20k development samples saved to `DATA/SAMPLE/`.
- Produced EDA visualizations covering rating distributions, temporal patterns, ingredient frequency, tag distributions, nutritional correlations, and user activity profiles (saved to `RESULTS/FIGURES/`).
- Authored Week 1 technical report and lecture notes (`REPORTS_AND_DOCS/`).

### Week 2 — Preprocessing and Feature Engineering (Complete)

**Preprocessing Pipeline (`SRC/data_preprocessing.py`, `NOTEBOOKS/03_DATA_PREPROCESSING.ipynb`)**

- Removed duplicate interactions by retaining the most recent review per user-recipe pair to prevent data leakage.
- Mapped explicit ratings (0–5 scale) to implicit binary feedback labels (`liked = 1` for ratings 4–5, `liked = 0` for ratings 1–3, `liked = NaN` for rating 0).
- Applied iterative 5-core filtering on the user-item bipartite interaction graph until convergence:
  - Raw sparsity: **99.99784%** → Post-filter sparsity: **99.92437%**
  - Retained: **17,813 users**, **41,240 recipes**, **555,618 interactions**
- Applied temporal user-wise 80:20 split (chronological, per-user) to prevent future-leakage:
  - `train_interactions.csv`: **437,558 interactions**
  - `test_interactions.csv`: **118,060 interactions**
  - Zero overlap between splits; every user has at least 4 training interactions guaranteed by the 5-core filter.

**Feature Engineering (`NOTEBOOKS/04_FEATURE_ENGINEERING.ipynb`)**

Engineered six health-aware features for **40,968 recipes** (post outlier removal):

| Feature | Description |
| --- | --- |
| `calorie_level` | Categorical: Low (<300 kcal), Medium (300–600 kcal), High (>600 kcal) |
| `protein_level` | Categorical: Low (<5g), Medium (5–15g), High (>15g) |
| `fat_level` | Categorical: Low (<10g), Medium (10–25g), High (>25g) |
| `ingredient_count` | Integer count of ingredients per recipe |
| `preparation_complexity` | Low (≤30 min AND ≤5 steps), High (>60 min OR >12 steps), Medium otherwise |
| `health_score` | Continuous score [0.0–10.0] derived from WHO dietary density guidelines across 5 macronutrient components |

WHO Health Score components (each capped at 2.0 points, total = 10.0):
1. Saturated fat energy contribution (threshold: <10% of total energy)
2. Free sugar energy contribution (threshold: <10% of total energy)
3. Sodium density (threshold: <1.0 mg/kcal)
4. Total fat energy contribution (threshold: <30% of total energy)
5. Protein energy contribution (adequacy threshold: ≥10% of total energy)

Health score range across dataset: **0.91 to 10.00** — zero missing values.

---

## 5. Team Core Contributions

The research roles are divided into three specific sub-systems:

- **Data and Sequential Recommendation:** Responsible for database connections, dataset conversions, user interaction sequence creation, train-test splits, and advanced sequential recommenders (GRU4Rec / SASRec).
- **Baseline Recommenders and Metrics:** Responsible for popularity, content-based, and matrix-factorization baseline models, evaluation ranking calculations, comparison tables, and offline simulation.
- **LLM/RAG and Health-Aware Optimization:** Responsible for building the knowledge base, vector index mapping, prompt design, health scoring, multi-objective trade-offs, and NLI-based faithfulness checks.

---

## 6. Getting Started

### 6.1 Prerequisites
Python 3.10+ is required. Verify that Python is on your PATH.

### 6.2 Installation
Clone this repository and install all dependencies:
```bash
pip install -r requirements.txt
```

### 6.3 Dataset Placement
The `DATA/` directory is not tracked by Git. Download the datasets and place raw files as follows:
1. **Food.com Recipes and Reviews** dataset:
   - `DATA/RAW/RAW_recipes.csv`
   - `DATA/RAW/RAW_interactions.csv`
2. **RecipeNLG** dataset:
   - `DATA/RAW/recipenlg.parquet`

### 6.4 Running the Notebooks
Execute notebooks in sequential order to reproduce the full pipeline:
```bash
jupyter notebook
```

| Notebook | Purpose |
| --- | --- |
| `NOTEBOOKS/01_DATASET_LOADING.ipynb` | Schema validation, memory profiling, and sample generation |
| `NOTEBOOKS/02_FOOD_DATASET_EDA.ipynb` | Exploratory analysis and figure generation |
| `NOTEBOOKS/03_DATA_PREPROCESSING.ipynb` | K-core filtering, implicit labels, and temporal split output |
| `NOTEBOOKS/04_FEATURE_ENGINEERING.ipynb` | WHO health score and categorical feature computation |
