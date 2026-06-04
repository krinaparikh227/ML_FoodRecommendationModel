# Week 1 Literature and Concept Notes: GroundedNutriRec Framework

This document contains the foundational literature review and concept notes for the **GroundedNutriRec** framework. It covers the core concepts of recommender systems, food recommendation, health-aware recommendation, explainable recommendation, Retrieval-Augmented Generation (RAG), and LLM hallucinations in explanations.

---

## 1. Recommender Systems

### Brief Introduction
Recommender systems are algorithms designed to suggest relevant items to users (such as movies, products, or food) by predicting their preferences based on historical interactions and user-item metadata.

### Detailed Explanation
Recommender systems generally fall into three main classes:
- **Collaborative Filtering**: Suggests items based on the similarity between users or items using interaction history (implicit or explicit feedback). It operates on the principle that users who agreed in the past will agree in the future.
- **Content-Based Filtering**: Recommends items similar to those a user liked in the past, based on item features (e.g., ingredients, category, cook time).
- **Hybrid Systems**: Combine collaborative and content-based approaches to leverage the strengths of both and mitigate cold-start issues.

Modern systems use deep learning (such as autoencoders or graph neural networks) and sequence modeling to capture complex interaction patterns.

### Examples
- Collaborative filtering using Singular Value Decomposition (SVD) to factorize a user-item rating matrix.
- Content-based filtering using TF-IDF vectors of recipe descriptions to find similar food items.

### Advantages
- Increases user engagement and retention by personalizing the user experience.
- Helps users discover new items from a massive catalog (long-tail items).
- Automates decision-making processes, reducing cognitive load for users.

### Disadvantages
- Subject to the **Cold-Start Problem** where new users or new items lack sufficient interaction data to make accurate predictions.
- Often suffers from data sparsity, as most users interact with only a tiny fraction of the total catalog.
- Can create filter bubbles, trapping users in narrow interest loops.

### Use Cases
- E-commerce platforms suggesting related products.
- Streaming services recommending music or movies.
- Food delivery apps suggesting restaurants or dishes.

### Limitations
- Cannot make high-quality predictions without historical data or rich metadata.
- Struggles in highly dynamic catalogs where items change rapidly.
- Ineffective when user preferences change suddenly and unpredictably.

---

## 2. Food Recommendation Systems

### Brief Introduction
Food recommendation systems are specialized recommender systems that suggest recipes, meals, or food products based on user dietary preferences, culinary history, and ingredient availability.

### Detailed Explanation
Unlike standard recommenders, food recommendation must capture:
- **Culinary Compatibility**: Matching ingredients that taste good together.
- **Session-Based Consumption**: Modeling user food preferences that change based on the time of day, season, or weekday vs. weekend.
- **Recipe Structure**: Parsing hierarchical text (title, ingredients, instructions) and converting it into structured vectors.

Food recommendation often relies on datasets like Food.com (recipes and reviews) or RecipeNLG to train content representations and collaborative models.

### Examples
- Recommending a chicken recipe based on a user's historical preference for Asian cuisine and chicken breast.
- Predicting the next meal in a sequence (e.g., recommending breakfast after detecting previous dinner interactions).

### Advantages
- Helps users plan meals and discover diverse culinary recipes.
- Minimizes food waste by recommending recipes based on available ingredients.
- Enhances cooking engagement by suggesting recipes matching the user's skill level.

### Disadvantages
- High difficulty in modeling the subjective nature of taste and flavor profiles.
- Standard recommender metrics (like accuracy) fail to capture physical consumption constraints (e.g., ingredient availability).
- Multi-ingredient interaction modeling requires complex NLP processing.

### Use Cases
- Recipe discovery websites and meal planning platforms.
- Smart kitchen appliances suggesting cooking routines.
- Grocery shopping assistants recommending complementary ingredients.

### Limitations
- Cannot verify if the user actually possesses the cooking tools or skill required for the recipe.
- Cannot dynamically predict ingredient freshness or quality.
- Cannot easily adapt if the user is cooking for a group with diverse preferences.

---

## 3. Health-Aware Recommendation Systems

### Brief Introduction
Health-aware recommendation systems integrate nutritional guidelines, physiological goals, and user health constraints to suggest foods that promote overall well-being.

### Detailed Explanation
These systems modify traditional utility functions by introducing **multi-objective optimization**. The recommendation score is a balance between user preference (personalization) and nutritional suitability (healthiness). Health suitability is evaluated using standards like:
- **Macronutrient Balance**: Target ratios of proteins, carbohydrates, and fats.
- **FSA/WHO Nutritional Scores**: Penalizing high levels of saturated fat, sugar, and sodium.
- **Dietary Constraints**: Allergies, diabetes management, or low-cholesterol targets.

### Examples
- Ranking recipes using a weighted combination of collaborative filtering scores and USDA nutrition-profile scores.
- Recommending low-sodium, high-protein meals to a user with hypertension who wants to build muscle.

### Advantages
- Proactively supports users in managing chronic illnesses (e.g., diabetes, cardiovascular disease).
- Encourages healthier eating habits at scale.
- Tailors food recommendations to active fitness goals.

### Disadvantages
- Health-promoting items may conflict directly with user taste preferences, leading to low adoption rates.
- Generating accurate health scores requires complete and structured nutrition metadata.
- Designing generalized health formulas is difficult, as nutritional needs vary widely based on age, weight, and activity.

### Use Cases
- Clinical dietary assistants for patients.
- Fitness tracker integrations recommending post-workout meals.
- School or hospital cafeteria menu planning.

### Limitations
- Cannot replace professional medical advice or personalized clinical nutrition assessments.
- Cannot verify user adherence to recommended diets.
- Struggles when user health profiles contain conflicting requirements (e.g., kidney disease and diabetes).

---

## 4. Explainable Recommendation Systems

### Brief Introduction
Explainable recommendation systems provide transparent reasons or justifications alongside recommended items, helping users understand why a particular suggestion was made.

### Detailed Explanation
Explanations in recommenders serve to improve user trust, transparency, and decision-making efficiency. They can be:
- **Feature-Based**: "Recommended because it contains chicken breast, which you cook frequently."
- **User-Based**: "People who liked X also liked Y."
- **Natural Language**: LLM-generated justifications explaining the alignment between the item's nutrition profile and the user's active health goals.

The GroundedNutriRec framework proposes using retrieval-augmented LLMs to generate explanations grounded directly in recipe evidence.

### Examples
- Displaying a highlight: "This recipe matches your goal of consuming under 400 calories and provides 25g of protein."
- Highlighting shared tags between user history and the recommended dish.

### Advantages
- Enhances user trust and system transparency.
- Helps users make informed choices rapidly by highlighting key features.
- Improves user satisfaction even when the recommendation itself is slightly suboptimal.

### Disadvantages
- Generating personalized natural language explanations is computationally expensive.
- Poorly constructed explanations can confuse users or expose system vulnerabilities.
- Explanation generation must be closely audited to prevent misleading justifications.

### Use Cases
- Interactive dietary apps explaining meal compositions.
- E-commerce recommendation blocks stating "Why recommended?".
- Medical decision support systems justifying diagnostic choices.

### Limitations
- Cannot force the user to read or act on the explanation.
- Subject to explanation bias, where persuasive text masks poor recommendation quality.
- Cannot easily verify if the user's subjective interpretation matches the system's intent.

---

## 5. Retrieval-Augmented Generation (RAG)

### Brief Introduction
Retrieval-Augmented Generation (RAG) is a framework that optimizes LLM outputs by querying an external knowledge base of verified documents before generating a response.

### Detailed Explanation
RAG separates knowledge storage from the language model's parametric memory. The workflow consists of:
1. **Indexing**: Converting document pages (e.g., recipes, nutrition guides) into vector embeddings and storing them in a database (like FAISS or ChromaDB).
2. **Retrieval**: Embedding the user query and retrieving the top-K most semantically similar documents from the database.
3. **Generation**: Appending the retrieved document context to the user prompt and instructing the LLM to generate the final response restricted strictly to the provided context.

This makes RAG highly effective for explainable food recommendation because the generated explanations are anchored in actual recipe facts.

### Examples
- A system querying a FAISS vector index of recipes using a user's preference query, retrieving the exact ingredients and steps of "Bourbon Chicken", and feeding them to an LLM to explain why it is a suitable high-protein meal.

### Advantages
- Dramatically reduces hallucinations by grounding the LLM in real-world reference documents.
- Allows real-time knowledge base updates without requiring expensive model fine-tuning.
- Provides clear auditability and citation capabilities for generated text.

### Disadvantages
- High latency due to the two-step process of vector search followed by LLM inference.
- Retrieval quality is heavily dependent on embedding model quality and document chunking strategies.
- Increased infrastructure complexity (vector databases, semantic indices).

### Use Cases
- Domain-specific QA assistants (legal, medical, or culinary).
- Document search engines with natural language summarization.
- Fact-grounded explanation generators for recommendation systems.

### Limitations
- Cannot generate accurate answers if the external database does not contain the required information (retrieval failure).
- Struggles when reasoning across multiple highly disconnected documents.
- Performance degrades if the document database contains contradictory or noisy information.

---

## 6. LLM Hallucinations in Explanations

### Brief Introduction
LLM hallucinations refer to instances where a generative model produces outputs that are grammatically correct but factually incorrect, fabricated, or unsupported by input context.

### Detailed Explanation
In explainable recommender systems, hallucinations represent a severe risk (e.g., stating a recipe is allergen-free or low-sugar when it is not). Hallucinations are driven by:
- **Parametric Bias**: The model relying on pre-trained associations rather than provided context.
- **Attention Over-smoothing**: The model generating text that fits language patterns but ignores source constraints.

GroundedNutriRec implements a **Claim-Level Faithfulness Verification** module to extract generated claims and verify them mathematically against the source evidence, computing a Faithfulness Score:
$$\text{Faithfulness Score} = \frac{\text{Supported Claims}}{\text{Total Claims}}$$

### Examples
- An LLM stating: "This banana bread recipe is excellent for diabetics because it is sugar-free," when the recipe actually contains 30g of granulated sugar.
- A model inventing ingredients or nutritional values that are not present in the database.

### Advantages
- Studying hallucinations enables the development of verification frameworks that act as safety guardrails.
- Quantifying hallucination rates provides objective metrics for evaluating system safety and reliability.

### Disadvantages
- Fabricated claims can endanger user safety, particularly in health and dietary domains.
- Detecting hallucinations dynamically requires high computational overhead (running secondary NLI models).
- Can severely damage user trust in the recommender system.

### Use Cases
- Evaluating and benchmarking safety metrics for generative AI.
- Developing guardrail layers for medical, financial, or legal language applications.
- Evaluating the fidelity of automated text summarization tools.

### Limitations
- Verification systems cannot completely eliminate the risk of hallucinations; they can only detect and filter them.
- Determining if a claim is "partially supported" vs "unsupported" remains a complex linguistic challenge.
- Detection models themselves are subject to errors and classification bias.
