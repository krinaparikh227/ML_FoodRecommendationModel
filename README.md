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
├── DATA/                   # Ignored dataset directories
│   ├── RAW/                # Raw datasets (Food.com, RecipeNLG)
│   ├── PROCESSED/          # Preprocessed data and train-test splits
│   └── SAMPLE/             # 20k development samples for prototyping
├── DATASETS/               # Zip archives containing raw dataset sources
├── NOTEBOOKS/              # Jupyter notebooks for weekly tasks
├── REPORTS_AND_DOCS/       # Academic literature notes and report documents
├── RESULTS/                # Output artifacts and tables
│   ├── FIGURES/            # Distribution plots and charts
│   └── TABLES/             # Baseline performance evaluation metrics
├── SRC/                    # Source code package
│   ├── EVALUATION/         # Metrics (Precision@K, NDCG@K, MRR@K)
│   ├── RAG/                # FAISS indexing and explanation prompt logic
│   ├── RECOMMENDERS/       # Collaborative filtering and sequential baselines
│   ├── __init__.py         # Package initialization
│   ├── data_preprocessing.py # Shared loading and data parsing scripts
│   └── utils.py            # Logger and path configuration helpers
├── requirements.txt        # Python dependencies
└── README.md               # Repository documentation
```

---

## 3. Weekly Research Roadmap

The project is structured into a 6-week summer program to progress from dataset understanding to integrated prototype verification:

| Week | Goal | Deliverables |
| --- | --- | --- |
| **Week 1** | Foundation & Dataset Exploration | Literature notes, schema validations, and sampling (`01_DATASET_LOADING.ipynb`) |
| **Week 2** | Preprocessing & Feature Engineering | Normalization, implicit mapping, splits, and nutrient level calculations (`03_data_preprocessing.ipynb`, `04_feature_engineering.ipynb`) |
| **Week 3** | Baselines & Evaluation Metrics | Matrix factorization, similarity recommenders, and ranking metrics (`05_popularity_recommender.ipynb`, `06_content_based_recommender.ipynb`, `07_collaborative_filtering.ipynb`, `08_evaluation_metrics.ipynb`) |
| **Week 4** | Health-Aware & Multi-Objective Ranking | Food health scores, multi-objective ranking functions, and ablation weights (`09_health_score_generation.ipynb`, `10_multi_objective_ranking.ipynb`, `11_ablation_weight_analysis.ipynb`, `21_pareto_ranking.ipynb`) |
| **Week 5** | LLM/RAG & Faithfulness Check | Vector index pipelines, explanation generation, NLI claim verification (`12_food_knowledge_base_creation.ipynb`, `13_vector_search_rag.ipynb`, `14_llm_explanation_generation.ipynb`, `18_claim_extraction.ipynb`, `19_faithfulness_verification.ipynb`, `20_hallucination_analysis.ipynb`) |
| **Week 6** | System Integration & Demos | Sequential model implementations, offline A/B testing simulator, Streamlit interface, and final report (`app/streamlit_app.py`, `16_sequential_recommender_gru.ipynb`, `17_transformer_based_recommender.ipynb`, `23_ab_testing_simulator.ipynb`) |

---

## 4. Team Core Contributions

The research roles are divided into three specific sub-systems:

- **Data and Sequential Recommendation:** Responsible for database connections, dataset conversions, user interaction sequence creation, train-test splits, and advanced sequential recommenders (GRU4Rec / SASRec).
- **Baseline Recommenders and Metrics:** Responsible for popularity, content-based, and matrix-factorization baseline models, evaluation ranking calculations, comparison tables, and offline simulation.
- **LLM/RAG and Health-Aware Optimization:** Responsible for building the knowledge base, vector index mapping, prompt design, health scoring, multi-objective trade-offs, and NLI-based faithfulness checks.

---

## 5. Getting Started

### 5.1 Prerequisites
Python 3.10+ is required. Verify that Python is on your PATH.

### 5.2 Installation
Clone this repository and install all dependencies:
```bash
pip install -r requirements.txt
```

### 5.3 Dataset Placement
1. Download the **Food.com Recipes and Reviews** dataset and **RecipeNLG** dataset.
2. Place the raw files inside the ignored directory: `DATA/RAW/`
   - `DATA/RAW/RAW_recipes.csv`
   - `DATA/RAW/RAW_interactions.csv`
   - `DATA/RAW/recipenlg.parquet` (or raw CSV datasets)

### 5.4 Running the Notebooks
Execute the notebook cells locally to initialize the preprocessing pipelines:
```bash
jupyter notebook
```
Start by opening `NOTEBOOKS/01_DATASET_LOADING.ipynb` to verify the environment and generate initial sampled partitions.
