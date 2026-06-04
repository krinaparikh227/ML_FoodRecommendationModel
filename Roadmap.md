6-Week Summer Internship Roadmap
Internship Title: Development of a Health-Aware and Explainable Food Recommendation System Using Machine Learning and Retrieval-Augmented LLMs
Paper Title: GroundedNutriRec: Retrieval-Augmented Multi-Objective LLM Framework for Health-Aware and Explainable Food Recommendation
1. Internship Objective
Primary goal: Prepare you to continue toward a Q1-journal-oriented research work on LLM-based food recommendation after the 6-week internship. The internship will not complete the full research paper, but it will produce the core pipeline, baseline results, documentation, and advanced task prototypes needed for the next 6 months.
2. Student Role Division
3. Recommended Datasets
4. Six-Week Roadmap
Week 1 - Foundation, Problem Understanding, and Dataset Exploration
Goal: Students understand the research problem, basics of recommender systems, and the structure of the selected food recommendation dataset.
Week 2 - Data Cleaning, Preprocessing, and Feature Engineering
Goal: Convert raw food interaction data into clean train/test data and engineered food metadata features.
Example implicit feedback label:
rating >= 4 -> liked = 1
rating < 4  -> liked = 0
Week 3 - Recommendation Baseline Models and Metrics
Goal: Implement baseline models that will later be used as comparative baselines in the research paper.
Week 4 - Health-Aware and Multi-Objective Food Ranking
Goal: Move from ordinary recommendation to the proposed research direction by balancing preference, health, popularity, diversity, and preparation time.
Initial multi-objective score:
FinalScore = alpha1 * PreferenceScore + alpha2 * HealthScore + alpha3 * PopularityScore + alpha4 * DiversityScore + alpha5 * PreparationTimeScore
Week 5 - LLM/RAG Explanation and Faithfulness Evaluation
Goal: Build a controlled RAG pipeline where the LLM explains recommendations only using retrieved food evidence.
Faithfulness Score = Supported Claims / Total Claims
Hallucination Rate = Unsupported Claims / Total Claims
Week 6 - Integration, Advanced Models, A/B Simulation, and Final Reporting
Goal: Integrate the pipeline into a working prototype, complete advanced technical tasks, and prepare research-ready documentation.
5. Additional Tasks
These tasks are intentionally challenging. They are designed to push you beyond basic implementation and prepare for publishable research work.
Task A - Sequential Food Recommendation Using GRU4Rec/SASRec
Objective: Predict the next food item based on user interaction history. This adds temporal behavior modeling to the recommender system.
Task B - Claim-Level Faithfulness Verification for RAG Explanations
Objective: Verify whether every generated explanation claim is supported by retrieved evidence. This is a strong LLM research contribution because it directly addresses hallucination.
Task C - Pareto-Based Multi-Objective Ranking
Objective: Replace fixed weighted ranking with Pareto-aware ranking to balance preference, health, preparation time, diversity, and popularity.
Task D - Offline A/B Testing Simulator (optional)
Objective: Simulate how two recommendation strategies would perform offline using test interactions and multi-objective metrics.
6. Integration of Advanced Tasks into Weekly Plan
7. Final Repository Structure
GroundedNutriRec/
|-- data/
|   |-- raw/
|   |-- processed/
|   |-- sample/
|-- notebooks/
|   |-- 01_dataset_loading.ipynb
|   |-- 02_food_dataset_eda.ipynb
|   |-- 03_data_preprocessing.ipynb
|   |-- 04_feature_engineering.ipynb
|   |-- 05_popularity_recommender.ipynb
|   |-- 06_content_based_recommender.ipynb
|   |-- 07_collaborative_filtering.ipynb
|   |-- 08_evaluation_metrics.ipynb
|   |-- 09_health_score_generation.ipynb
|   |-- 10_multi_objective_ranking.ipynb
|   |-- 11_ablation_weight_analysis.ipynb
|   |-- 12_food_knowledge_base_creation.ipynb
|   |-- 13_vector_search_rag.ipynb
|   |-- 14_llm_explanation_generation.ipynb
|   |-- 15_faithfulness_check.ipynb
|   |-- 16_sequential_recommender_gru.ipynb
|   |-- 17_transformer_based_recommender.ipynb
|   |-- 18_claim_extraction.ipynb
|   |-- 19_faithfulness_verification.ipynb
|   |-- 20_hallucination_analysis.ipynb
|   |-- 21_pareto_ranking.ipynb
|   |-- 22_multi_objective_comparison.ipynb
|   |-- 23_ab_testing_simulator.ipynb
|-- src/
|   |-- data_preprocessing.py
|   |-- recommenders/
|   |-- rag/
|   |-- evaluation/
|   |-- utils.py
|-- app/
|   |-- streamlit_app.py
|-- results/
|   |-- tables/
|   |-- figures/
|-- reports/
|-- requirements.txt
|-- README.md
|-- research_plan_next_6_months.md
8. Final Internship Deliverables Checklist
9. Final Research Readiness After 6 Weeks
After the internship, you all are ready to continue the following 6-month research tasks: 
advanced sequential recommendation models
stronger LLM-based intent extraction
improved RAG and explanation verification
experiments on Food.com and Amazon Grocery
multi-objective optimization and Pareto ranking
human evaluation of explanations
ablation study
paper writing and Q1 journal targeting
Outcome Area | Expected Capability After 6 Weeks
Recommendation Systems | Understand users, items, interactions, top-K recommendation, candidate generation, ranking, and evaluation.
Dataset Handling | Load, clean, analyze, and preprocess Food.com or similar benchmark food recommendation datasets.
Baseline Models | Implement popularity-based, rating-based, content-based, and collaborative filtering recommenders.
Research Module | Implement health-aware and multi-objective food ranking.
LLM/RAG Module | Build a basic RAG-based explanation system using retrieved food evidence.
Advanced Tasks | Attempt sequential recommendation, claim-level faithfulness evaluation, Pareto ranking, and offline A/B testing simulation.
Documentation | Maintain GitHub repository, notebooks, weekly reports, final internship report, and result tables.
Student | Role | Main Responsibility | Advanced Challenge
Student 1 | Data + Sequential Recommendation Lead | Dataset collection, cleaning, EDA, user sequence creation, train-test split, preprocessing pipeline. | GRU4Rec or Transformer/SASRec sequential recommender.
Student 2 | Baseline Recommendation + Evaluation Lead | Popularity, rating-based, content-based, collaborative filtering, ranking metrics, comparison tables. | Offline A/B testing simulator and model comparison.
Student 3 | LLM/RAG + Multi-Objective Ranking Lead | Food knowledge base, vector search, RAG explanations, health scoring, multi-objective ranking. | Claim-level faithfulness evaluator and Pareto ranking.
Dataset | Primary Use | Notes
Food.com Recipes and Interactions | Main benchmark for food recommendation, recipe metadata, user interactions, reviews, ingredients, and nutrition-aware ranking. | Use this as the primary internship dataset.
Food.com Recipes and Reviews | Richer recipe metadata, nutrition fields, instructions, and recipe text for RAG. | Use if students need more detailed knowledge-base evidence.
Amazon Reviews 2023 - Grocery and Gourmet Food | Industrial food-product recommendation and review-grounded explanation. | Use in the 6-month research phase after Food.com baseline is stable.
MealRec+ | Meal-level recommendation and bundle-style food planning. | Optional advanced dataset.
Ele.me / Tianchi Food Delivery | Operational/time-aware recommendation proxy. | Optional for queue/delivery-time-aware extension.
Task | Details | Deliverable
Literature and Concept Notes | Study recommender systems, food recommendation, health-aware recommendation, explainable recommendation, LLM, RAG, and hallucination. | Week1_Literature_Notes.docx
Dataset Loading | Download and load Food.com recipes and interactions. Identify columns, data types, and sample records. | 01_dataset_loading.ipynb
Basic EDA | Find number of users, recipes, interactions, rating distribution, top recipes, missing values, recipe categories, and nutrition field availability. | 02_food_dataset_eda.ipynb
Friday Review | Each student presents what they understood, dataset challenges, and screenshots of EDA results. | Week 1 presentation + GitHub commits
Task | Details | Deliverable
Clean Interaction Data | Remove duplicate interactions, handle missing ratings, normalize ratings, filter sparse users/items, create implicit feedback label. | 03_data_preprocessing.ipynb
Food Metadata Table | Prepare item_id, recipe name, ingredients, calories, protein, fat, carbs, category, preparation time, and review text. | food_metadata_clean.csv
Nutrition Features | Create calorie_level, protein_level, fat_level, ingredient_count, health_score, and preparation_complexity. | 04_feature_engineering.ipynb
Train-Test Split | Create user-wise 80:20 split for top-K recommendation evaluation. | train_interactions.csv, test_interactions.csv
Friday Review | Validate cleaned data, explain feature engineering logic, and commit processed files. | Week 2 report
Model/Task | Implementation Details | Implementation Details | Implementation Details | Deliverable
Popularity-Based Recommender | Recommend most interacted and highly rated food items. | Recommend most interacted and highly rated food items. | Recommend most interacted and highly rated food items. | 05_popularity_recommender.ipynb
Rating-Based Recommender | Recommend highest-rated items with minimum interaction threshold. | Recommend highest-rated items with minimum interaction threshold. | Recommend highest-rated items with minimum interaction threshold. | 05_popularity_recommender.ipynb
Content-Based Recommender | Use TF-IDF or Sentence Transformer embeddings on recipe name, ingredients, and category. | Use TF-IDF or Sentence Transformer embeddings on recipe name, ingredients, and category. | Use TF-IDF or Sentence Transformer embeddings on recipe name, ingredients, and category. | 06_content_based_recommender.ipynb
Collaborative Filtering | Use user-item matrix with cosine similarity, Surprise SVD, or implicit ALS. | Use user-item matrix with cosine similarity, Surprise SVD, or implicit ALS. | Use user-item matrix with cosine similarity, Surprise SVD, or implicit ALS. | 07_collaborative_filtering.ipynb
Evaluation Metrics | Implement Precision@K, Recall@K, HitRate@K, NDCG@K, and MRR@K. | Implement Precision@K, Recall@K, HitRate@K, NDCG@K, and MRR@K. | Implement Precision@K, Recall@K, HitRate@K, NDCG@K, and MRR@K. | 08_evaluation_metrics.ipynb
Model | Precision@5 | Recall@5 | HitRate@10 | NDCG@10
Popularity | To be filled | To be filled | To be filled | To be filled
Rating-based | To be filled | To be filled | To be filled | To be filled
Content-based | To be filled | To be filled | To be filled | To be filled
Collaborative filtering | To be filled | To be filled | To be filled | To be filled
Task | Task | Details | Details | Deliverable | Deliverable
Preference Score | Preference Score | Use collaborative filtering predicted score or content-based similarity score. | Use collaborative filtering predicted score or content-based similarity score. | 10_multi_objective_ranking.ipynb | 10_multi_objective_ranking.ipynb
Health Score | Health Score | Use normalized protein, calorie, fat, and ingredient-diversity features. | Use normalized protein, calorie, fat, and ingredient-diversity features. | 09_health_score_generation.ipynb | 09_health_score_generation.ipynb
Preparation-Time Score | Preparation-Time Score | Use recipe preparation time as a proxy for quickness. | Use recipe preparation time as a proxy for quickness. | 10_multi_objective_ranking.ipynb | 10_multi_objective_ranking.ipynb
Weight Ablation | Weight Ablation | Evaluate multiple combinations of preference, health, popularity, and time weights. | Evaluate multiple combinations of preference, health, popularity, and time weights. | 11_ablation_weight_analysis.ipynb | 11_ablation_weight_analysis.ipynb
Advanced Task Start | Advanced Task Start | Begin Pareto ranking implementation as a stronger alternative to fixed weighted scoring. | Begin Pareto ranking implementation as a stronger alternative to fixed weighted scoring. | 21_pareto_ranking.ipynb | 21_pareto_ranking.ipynb
Variant | Precision@10 | Recall@10 | Avg Health Score | Avg Prep Time | Diversity
Preference only | To be filled | To be filled | To be filled | To be filled | To be filled
Preference + Health | To be filled | To be filled | To be filled | To be filled | To be filled
Preference + Health + Time | To be filled | To be filled | To be filled | To be filled | To be filled
Full Multi-Objective | To be filled | To be filled | To be filled | To be filled | To be filled
Task | Details | Deliverable
Food Knowledge Base | Create one document per recipe/food item containing name, ingredients, calories, protein, fat, prep time, rating, and review summary. | 12_food_knowledge_base_creation.ipynb
Vector Search | Build FAISS, ChromaDB, or Qdrant vector store using SentenceTransformer embeddings. | 13_vector_search_rag.ipynb
RAG Explanation | Generate explanations based only on retrieved item evidence. | 14_llm_explanation_generation.ipynb
Claim-Level Verification | Split explanations into claims and check whether claims are supported by retrieved evidence. | 18_claim_extraction.ipynb, 19_faithfulness_verification.ipynb
Hallucination Analysis | Compute faithfulness score and hallucination rate. | 20_hallucination_analysis.ipynb
Task | Details | Deliverable
Streamlit Demo | Build a simple app where user selects healthy/high-protein/quick/popular/balanced preferences and receives top-5 food recommendations with explanation. | app/streamlit_app.py
Sequential Recommender | Create user sequences and implement Markov Chain, GRU4Rec, or Transformer/SASRec depending on student capability. | 16_sequential_recommender_gru.ipynb, 17_transformer_based_recommender.ipynb
Offline A/B Testing Simulator | Compare popularity vs proposed model, preference-only vs health-aware ranking, LLM-only vs RAG explanation. | 23_ab_testing_simulator.ipynb
Final Results | Create baseline result tables, ablation tables, faithfulness tables, and sample explanations. | results/tables, results/figures
Final Documentation | Prepare final internship report, README, research gap document, and next 6-month roadmap. | final_internship_report.docx, README.md, research_plan_next_6_months.md
Level | Model | Model | Expected Work
Medium | Markov Chain | Markov Chain | Estimate next item using transition probabilities between food items.
High | GRU4Rec | GRU4Rec | Train a GRU-based neural recommender on user item sequences.
Very High | SASRec / Transformer | SASRec / Transformer | Use self-attention to model user sequence patterns and predict next item.
Implementation Step | Implementation Step | Description | Description
Sequence Creation | Sequence Creation | Sort interactions by user and timestamp. | Sort interactions by user and timestamp.
Input-Target Pairs | Input-Target Pairs | Generate examples such as [Poha, Tea, Sandwich] -> Paneer Roll. | Generate examples such as [Poha, Tea, Sandwich] -> Paneer Roll.
Model Training | Model Training | Train GRU/Transformer model for next-item prediction. | Train GRU/Transformer model for next-item prediction.
Evaluation | Evaluation | Use HitRate@10, NDCG@10, and MRR@10. | Use HitRate@10, NDCG@10, and MRR@10.
Difficulty Level | Method | Method
Basic | Rule-based keyword and threshold verification. | Rule-based keyword and threshold verification.
Intermediate | Sentence embedding similarity between claim and evidence. | Sentence embedding similarity between claim and evidence.
Advanced | Natural Language Inference model for entailment/contradiction checking. | Natural Language Inference model for entailment/contradiction checking.
Claim | Evidence | Supported?
High in protein | Protein = 14g | Yes
Costs less than Rs. 80 | Price = Rs. 70 | Yes
Low fat | Fat = 18g | No
Prepared quickly | Preparation time = 8 minutes | Yes
Experiment | Description
E1 | Preference-only ranking.
E2 | Fixed weighted multi-objective ranking.
E3 | Pareto-front ranking.
E4 | Personalized Pareto ranking based on user intent.
Metric | Purpose
Precision@K | Measures relevance of recommendations.
Average Health Score | Measures nutritional quality.
Average Preparation Time | Measures quickness.
Diversity@K | Measures variety in recommended food items.
Coverage | Measures how many unique items are recommended.
Trade-off Score | Measures balance among objectives.
Strategy A | Strategy B | Comparison Goal
Popularity-based | Proposed multi-objective ranking | Measure whether proposed ranking improves relevance and health score.
Preference-only | Health-aware ranking | Measure health benefit versus accuracy trade-off.
LLM-only explanation | RAG-grounded explanation | Measure hallucination reduction and faithfulness gain.
Week | Core Work | Advanced Challenge
Week 1 | Dataset understanding and EDA | Read one paper/blog on sequential recommendation or explainable recommendation.
Week 2 | Preprocessing and feature engineering | Prepare timestamp-sorted interactions for sequential recommendation.
Week 3 | Baseline recommender implementation | Prepare baseline tables for later A/B testing.
Week 4 | Health-aware multi-objective ranking | Start Pareto-front ranking implementation.
Week 5 | RAG explanation generation | Implement claim extraction and faithfulness verification.
Week 6 | Integration and final reporting | Complete sequential model, A/B simulator, and final comparison tables.
Deliverable
Dataset loading and EDA notebooks
Cleaned train/test data and metadata table
Popularity, rating, content-based, and CF recommenders
Top-K evaluation metrics
Health score and multi-objective ranking
Pareto ranking and comparison
Food knowledge base and vector search
RAG explanations and faithfulness checking
Sequential recommender
Offline A/B testing simulator
Streamlit demo
Final report and GitHub README