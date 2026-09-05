# 13 — Vector Search RAG Pipeline

**Project:** GroundedNutriRec  
**Scope:** Build a controlled RAG (Retrieval-Augmented Generation) pipeline where the LLM explains food recommendations **only** using retrieved evidence from the Food Knowledge Base.  

## Objectives
1. Load the food knowledge base created in Notebook 12.
2. Generate dense embeddings using **SentenceTransformers** (`all-MiniLM-L6-v2`).
3. Build a **FAISS** vector index for fast approximate nearest-neighbour search.
4. Implement a **retrieval** function that, given a user query, returns the top-k most relevant food documents.
5. Construct a **grounded prompt** that feeds retrieved evidence to an LLM and instructs it to answer **only** from the provided context.
6. Demonstrate end-to-end RAG with sample queries.
7. Persist the FAISS index for reuse.

## 1. Install Dependencies

Install the required packages if not already present.


```python
# Install required packages (run once)
import subprocess, sys

def install(package):
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', package])

for pkg in ['sentence-transformers', 'faiss-cpu', 'transformers', 'google-genai']:
    try:
        __import__(pkg.replace('-', '_'))
    except ImportError:
        print(f'Installing {pkg}...')
        install(pkg)

print('All dependencies ready.')
```

    Installing faiss-cpu...
    

    Installing google-genai...
    

    All dependencies ready.
    

## 2. Setup and Imports


```python
import pandas as pd
import numpy as np
import json
import faiss
import warnings
import pickle
import time
from pathlib import Path
from sentence_transformers import SentenceTransformer
from transformers import pipeline

warnings.filterwarnings('ignore')

# -- Paths --
KB_CSV_PATH    = Path('dataset/archive_3/food_knowledge_base.csv')
FAISS_DIR      = Path('dataset/archive_3/faiss_index')
FAISS_DIR.mkdir(parents=True, exist_ok=True)
INDEX_PATH     = FAISS_DIR / 'food_kb.index'
METADATA_PATH  = FAISS_DIR / 'food_kb_meta.pkl'

# -- Embedding Model --
EMBED_MODEL_NAME = 'all-MiniLM-L6-v2'   # 384-dim, fast & effective

# -- HuggingFace Generation Model --
print('Loading local LLM (TinyLlama)...')
llm_pipeline = pipeline('text-generation', model='TinyLlama/TinyLlama-1.1B-Chat-v1.0', max_new_tokens=256, temperature=0.3, do_sample=True)

print(f'Knowledge base  : {KB_CSV_PATH}')
print(f'FAISS index dir : {FAISS_DIR}')
print(f'Embedding model : {EMBED_MODEL_NAME}')
print(f'LLM             : TinyLlama-1.1B (Local)')
print('Setup complete.')
```

    Loading local LLM (TinyLlama)...
    


    config.json:   0%|          | 0.00/608 [00:00<?, ?B/s]



    model.safetensors:   0%|          | 0.00/2.20G [00:00<?, ?B/s]



    Loading weights:   0%|          | 0/201 [00:00<?, ?it/s]



    generation_config.json:   0%|          | 0.00/124 [00:00<?, ?B/s]



    tokenizer_config.json:   0%|          | 0.00/1.29k [00:00<?, ?B/s]


    Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
    


    tokenizer.model:   0%|          | 0.00/500k [00:00<?, ?B/s]



    tokenizer.json:   0%|          | 0.00/1.84M [00:00<?, ?B/s]



    special_tokens_map.json:   0%|          | 0.00/551 [00:00<?, ?B/s]


    [transformers] Passing `generation_config` together with generation-related arguments=({'temperature', 'max_new_tokens', 'do_sample'}) is deprecated and will be removed in future versions. Please pass either a `generation_config` object OR all generation parameters explicitly, but not both.
    

    Knowledge base  : dataset\archive_3\food_knowledge_base.csv
    FAISS index dir : dataset\archive_3\faiss_index
    Embedding model : all-MiniLM-L6-v2
    LLM             : TinyLlama-1.1B (Local)
    Setup complete.
    

## 3. Load Food Knowledge Base


```python
print('Loading knowledge base...')
kb_df_full = pd.read_csv(KB_CSV_PATH)
print(f'  Full KB shape: {kb_df_full.shape}')

# Drop any rows with missing documents
kb_df_full = kb_df_full.dropna(subset=['document']).reset_index(drop=True)

# --- Sample top-rated recipes for CPU-friendly embedding ---
# Encoding 231K documents on CPU would take hours.
# We select the top 20,000 recipes by average rating (with at least 1 review)
# to build a high-quality, representative vector store.
SAMPLE_SIZE = 20_000

kb_df = (
    kb_df_full[kb_df_full['review_count'] > 0]
    .sort_values('avg_rating', ascending=False)
    .head(SAMPLE_SIZE)
    .reset_index(drop=True)
)

print(f'  Sampled {len(kb_df):,} top-rated recipes (from {len(kb_df_full):,} total)')
print(f'  Rating range: {kb_df["avg_rating"].min():.2f} - {kb_df["avg_rating"].max():.2f}')

documents = kb_df['document'].tolist()
print(f'\nSample document:')
print('=' * 60)
print(documents[0])
```

    Loading knowledge base...
    

      Full KB shape: (231637, 11)
    

      Sampled 20,000 top-rated recipes (from 231,637 total)
      Rating range: 5.00 - 5.00
    
    Sample document:
    ============================================================
    Recipe: zydeco ya ya deviled eggs
    Ingredients: hard-cooked eggs, mayonnaise, dijon mustard, salt-free cajun seasoning, tabasco sauce, salt, black pepper, fresh italian parsley
    Calories: 59.2 kcal | Protein: 6.0 %DV | Fat: 6.0 %DV
    Prep Time: 40 minutes
    Average Rating: 5.0/5 (5 reviews)
    Review Summary: I loved these eggs. How different, how good, and such a nice taste. I absolutely loved the use of Tabasco sauce here, and the Cajun seasoning. I halved this recipe easily and did reduce the mayo and used 2 tablespoons. Thank you for posting such a different and totally wonderful deviled egg to devour! Made for *Think Pink* Breast Cancer Awareness Month Fall 2008
    

## 4. Generate SentenceTransformer Embeddings

We encode every knowledge-base document into a dense vector using `all-MiniLM-L6-v2` (384 dimensions).  
This model provides a good trade-off between speed and semantic quality.

> **Note:** Encoding ~230 k documents may take several minutes on CPU. A GPU will speed this up significantly.


```python
print(f'Loading SentenceTransformer model: {EMBED_MODEL_NAME} ...')
model = SentenceTransformer(EMBED_MODEL_NAME)
print(f'  Embedding dimension: {model.get_sentence_embedding_dimension()}')

# ── Encode documents ──────────────────────────────────
BATCH_SIZE = 256

print(f'\nEncoding {len(documents):,} documents (batch_size={BATCH_SIZE})...')
start = time.time()

embeddings = model.encode(
    documents,
    batch_size=BATCH_SIZE,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True   # L2-normalise for cosine similarity via inner-product
)

elapsed = time.time() - start
print(f'\nEncoding complete in {elapsed:.1f}s')
print(f'Embeddings shape: {embeddings.shape}')
```

    Loading SentenceTransformer model: all-MiniLM-L6-v2 ...
    


    modules.json:   0%|          | 0.00/349 [00:00<?, ?B/s]



    config_sentence_transformers.json:   0%|          | 0.00/116 [00:00<?, ?B/s]



    README.md:   0%|          | 0.00/10.5k [00:00<?, ?B/s]



    sentence_bert_config.json:   0%|          | 0.00/53.0 [00:00<?, ?B/s]



    config.json:   0%|          | 0.00/612 [00:00<?, ?B/s]



    model.safetensors:   0%|          | 0.00/90.9M [00:00<?, ?B/s]



    Loading weights:   0%|          | 0/103 [00:00<?, ?it/s]



    tokenizer_config.json:   0%|          | 0.00/350 [00:00<?, ?B/s]



    vocab.txt:   0%|          | 0.00/232k [00:00<?, ?B/s]



    tokenizer.json:   0%|          | 0.00/466k [00:00<?, ?B/s]



    special_tokens_map.json:   0%|          | 0.00/112 [00:00<?, ?B/s]



    config.json:   0%|          | 0.00/190 [00:00<?, ?B/s]


      Embedding dimension: 384
    
    Encoding 20,000 documents (batch_size=256)...
    


    Batches:   0%|          | 0/79 [00:00<?, ?it/s]


    
    Encoding complete in 472.2s
    Embeddings shape: (20000, 384)
    

## 5. Build FAISS Vector Index

We use **FAISS IndexFlatIP** (Inner Product) because our embeddings are L2-normalised, so inner product equals cosine similarity.  
For very large collections, an approximate index (e.g. `IndexIVFFlat`) would be faster, but the flat index guarantees exact results.


```python
dim = embeddings.shape[1]
print(f'Building FAISS IndexFlatIP (dim={dim})...')

index = faiss.IndexFlatIP(dim)
index.add(embeddings.astype(np.float32))

print(f'  Vectors in index: {index.ntotal:,}')
print('FAISS index built successfully.')
```

    Building FAISS IndexFlatIP (dim=384)...
      Vectors in index: 20,000
    FAISS index built successfully.
    

## 6. Persist FAISS Index & Metadata


```python
# ── Save FAISS index ──
faiss.write_index(index, str(INDEX_PATH))
print(f'FAISS index saved -> {INDEX_PATH}')

# ── Save metadata (recipe_ids + documents) ──
metadata = {
    'recipe_ids': kb_df['recipe_id'].tolist(),
    'documents':  documents,
    'names':      kb_df['name'].tolist(),
}
with open(METADATA_PATH, 'wb') as f:
    pickle.dump(metadata, f)
print(f'Metadata saved   -> {METADATA_PATH}')
print('\n[OK] Index persisted.')
```

    FAISS index saved -> dataset\archive_3\faiss_index\food_kb.index
    Metadata saved   -> dataset\archive_3\faiss_index\food_kb_meta.pkl
    
    [OK] Index persisted.
    

## 7. Semantic Retrieval Function

Given a natural-language query, encode it and retrieve the **top-k** most similar documents from the FAISS index.


```python
def retrieve(query: str, model, index, metadata, top_k: int = 5):
    """
    Retrieve the top-k most relevant food documents for a given query.
    
    Parameters
    ----------
    query    : str  – natural-language user query
    model    : SentenceTransformer model
    index    : FAISS index
    metadata : dict with 'documents', 'recipe_ids', 'names'
    top_k    : int  – number of results to return
    
    Returns
    -------
    list[dict] – each dict has keys: rank, score, recipe_id, name, document
    """
    # Encode query
    q_emb = model.encode([query], normalize_embeddings=True).astype(np.float32)
    
    # Search
    scores, indices = index.search(q_emb, top_k)
    
    results = []
    for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), start=1):
        results.append({
            'rank':      rank,
            'score':     float(score),
            'recipe_id': metadata['recipe_ids'][idx],
            'name':      metadata['names'][idx],
            'document':  metadata['documents'][idx],
        })
    return results

print('Retrieval function defined. [OK]')
```

    Retrieval function defined. [OK]
    

## 8. Test Retrieval with Sample Queries


```python
test_queries = [
    'high protein low fat chicken recipe',
    'quick breakfast under 15 minutes',
    'healthy vegetarian pasta with low calories',
    'best rated chocolate dessert',
]

for query in test_queries:
    print(f'\n{"═" * 80}')
    print(f'QUERY: {query}')
    print(f'{"═" * 80}')
    results = retrieve(query, model, index, metadata, top_k=3)
    for r in results:
        print(f'\n  [{r["rank"]}] Score: {r["score"]:.4f}  |  {r["name"]}')
        # Show first 200 chars of document
        snippet = r['document'][:200].replace('\n', ' | ')
        print(f'      {snippet}...')
```

    
    ════════════════════════════════════════════════════════════════════════════════
    QUERY: high protein low fat chicken recipe
    ════════════════════════════════════════════════════════════════════════════════
    
      [1] Score: 0.6588  |  walnut   herb crusted chicken
          Recipe: walnut   herb crusted chicken | Ingredients: walnuts, plain breadcrumbs, dried basil leaves, dried rosemary leaves, flour, salt, egg white, chicken pieces | Calories: 560.9 kcal | Protein: 75.0 %D...
    
      [2] Score: 0.6297  |  three bean salad  high protein
          Recipe: three bean salad  high protein | Ingredients: cut green beans, red kidney beans, chickpeas, onion, fat-free italian salad dressing | Calories: 213.3 kcal | Protein: 21.0 %DV | Fat: 2.0 %DV | Prep Ti...
    
      [3] Score: 0.6284  |  the best chicken breasts
          Recipe: the best chicken breasts | Ingredients: boneless skinless chicken breast halves, turkey bacon, dried beef, 98% fat-free cream of chicken soup, fat free sour cream | Calories: 219.9 kcal | Protein:...
    
    ════════════════════════════════════════════════════════════════════════════════
    QUERY: quick breakfast under 15 minutes
    ════════════════════════════════════════════════════════════════════════════════
    
      [1] Score: 0.6042  |  yummy quick   healthy breakfast sandwich
          Recipe: yummy quick   healthy breakfast sandwich | Ingredients: bagels, bananas, cinnamon | Calories: 787.8 kcal | Protein: 49.0 %DV | Fat: 6.0 %DV | Prep Time: 2 minutes | Average Rating: 5.0/5 (1 reviews) | R...
    
      [2] Score: 0.5228  |  x lazy early riser crock pot breakfast
          Recipe: x lazy early riser crock pot breakfast | Ingredients: potatoes, onions, ham, sharp cheddar cheese, egg substitute, salt, pepper, dried mustard | Calories: 302.1 kcal | Protein: 64.0 %DV | Fat: 17....
    
      [3] Score: 0.5111  |  bacon   egg sandwiches  breakfast  lunch mmmmmmm
          Recipe: bacon   egg sandwiches  breakfast  lunch mmmmmmm | Ingredients: sour cream, bread, green onions, process american cheese, hard-boiled eggs, bacon, butter | Calories: 640.4 kcal | Protein: 38.0 %DV...
    
    ════════════════════════════════════════════════════════════════════════════════
    QUERY: healthy vegetarian pasta with low calories
    ════════════════════════════════════════════════════════════════════════════════
    

    
      [1] Score: 0.7065  |  vegan pasta casserole
          Recipe: vegan pasta casserole | Ingredients: flour, nutritional yeast flakes, vegetable broth, water, braggs liquid aminos, garlic powder, oregano, dried basil, paprika, tomato sauce, cooked pasta | Calor...
    
      [2] Score: 0.7008  |  all in one pot saucy pasta
          Recipe: all in one pot saucy pasta | Ingredients: extra lean ground beef, onion, rotini pasta, water, spaghetti sauce, fresh mushrooms, red pepper, part-skim mozzarella cheese | Calories: 696.0 kcal | Pro...
    
      [3] Score: 0.6919  |  vegetarian spaghetti for crock pot
          Recipe: vegetarian spaghetti for crock pot | Ingredients: spaghetti sauce mix, tomato sauce, water, eggplant, bell pepper, tomatoes, salt, spaghetti, mozzarella cheese | Calories: 265.2 kcal | Protein: 23...
    
    ════════════════════════════════════════════════════════════════════════════════
    QUERY: best rated chocolate dessert
    ════════════════════════════════════════════════════════════════════════════════
    
      [1] Score: 0.6286  |  the bestest chocolate cake ever  with chocolate frosting
          Recipe: the bestest chocolate cake ever  with chocolate frosting | Ingredients: sugar, flour, cocoa, baking powder, baking soda, salt, eggs, milk, vegetable oil, vanilla, water, butter, powdered sugar | C...
    
      [2] Score: 0.6222  |  ultimate hot chocolate
          Recipe: ultimate hot chocolate | Ingredients: heavy cream, milk, sweetened condensed milk, semi-sweet chocolate chips, cocoa powder | Calories: 434.6 kcal | Protein: 16.0 %DV | Fat: 46.0 %DV | Prep Time: 16...
    
      [3] Score: 0.6165  |  beautiful chocolate disks with seeds and fruits
          Recipe: beautiful chocolate disks with seeds and fruits | Ingredients: semi-sweet chocolate, milk chocolate, cinnamon, ginger, nutmeg, chili powder, dried cranberries, poppy seed, sunflower seeds, vanil...
    

## 9. Grounded RAG Prompt Construction

The key to a **controlled RAG** pipeline is the prompt template.  
We instruct the LLM to:
1. **Only** use the retrieved food evidence.
2. **Cite** which recipe(s) the recommendation is based on.
3. **Refuse** to answer if the evidence does not support the query.


```python
RAG_SYSTEM_PROMPT = """\
You are a nutrition-aware food recommendation assistant.
You MUST answer the user's question ONLY using the retrieved food evidence provided below.
Do NOT use any external knowledge or make up information.

Rules:
1. Base your answer strictly on the EVIDENCE section.
2. Cite the recipe name(s) you reference.
3. Include relevant nutritional information (calories, protein, fat) from the evidence.
4. If the evidence does not contain enough information to answer, say:
   "I don't have enough evidence to answer this question."
5. Keep your answer concise and helpful.
"""

def build_rag_prompt(query: str, retrieved_docs: list) -> str:
    """
    Build a grounded RAG prompt from the user query and retrieved documents.
    """
    evidence_block = '\n\n'.join(
        f'--- Evidence {doc["rank"]} (score: {doc["score"]:.4f}) ---\n{doc["document"]}'
        for doc in retrieved_docs
    )
    
    # TinyLlama chat format
    prompt = (
        f'<|system|>\n'
        f'{RAG_SYSTEM_PROMPT}\n'
        f'=== EVIDENCE ===\n'
        f'{evidence_block}</s>\n'
        f'<|user|>\n'
        f'{query}</s>\n'
        f'<|assistant|>\n'
    )
    return prompt


def generate_answer(query: str, retrieved_docs: list) -> str:
    """
    Generate a grounded answer using local TinyLlama.
    """
    prompt = build_rag_prompt(query, retrieved_docs)
    output = llm_pipeline(prompt, return_full_text=False)
    return output[0]['generated_text'].strip()

print('RAG prompt builder defined.')
print('Local LLM answer generator defined.')
```

    RAG prompt builder defined.
    Local LLM answer generator defined.
    

## 10. End-to-End RAG Demonstration

We demonstrate the full pipeline: **Query -> Retrieve -> Prompt -> Gemini LLM -> Grounded Answer**


```python
# -- Full pipeline demo --
demo_query = 'I want a high-protein, low-calorie meal that is quick to prepare'

print(f'USER QUERY: {demo_query}')
print(f'{chr(9472) * 80}\n')

# Step 1: Retrieve relevant documents
retrieved = retrieve(demo_query, model, index, metadata, top_k=5)
print(f'Retrieved {len(retrieved)} documents:\n')
for r in retrieved:
    print(f'  [{r["rank"]}] {r["name"]}  (score={r["score"]:.4f})')

# Step 2: Generate grounded answer using Gemini
print(f'\n{"=" * 80}')
print('GEMINI GROUNDED ANSWER:')
print(f'{"=" * 80}\n')

answer = generate_answer(demo_query, retrieved)
print(answer)
```

    USER QUERY: I want a high-protein, low-calorie meal that is quick to prepare
    ────────────────────────────────────────────────────────────────────────────────
    
    Retrieved 5 documents:
    
      [1] three bean salad  high protein  (score=0.6057)
      [2] 5 minute comfort food  (score=0.5642)
      [3] yummy quick   healthy breakfast sandwich  (score=0.5612)
      [4] beakfast on the run  (score=0.5600)
      [5] bacon   egg sandwiches  breakfast  lunch mmmmmmm  (score=0.5553)
    
    ================================================================================
    GEMINI GROUNDED ANSWER:
    ================================================================================
    
    

    [transformers] Ignoring clean_up_tokenization_spaces=True for BPE tokenizer LlamaTokenizer. The clean_up_tokenization post-processing step is designed for WordPiece tokenizers and is destructive for BPE (it strips spaces before punctuation). Set clean_up_tokenization_spaces=False to suppress this warning, or set clean_up_tokenization_spaces_for_bpe_even_though_it_will_corrupt_output=True to force cleanup anyway.
    

    Here's a recipe for a quick and easy protein-packed meal that is low in calories:
    
    Ingredients:
    - 1 cup cooked quinoa
    - 1 can black beans, drained and rinsed
    - 1 small red bell pepper, seeded and chopped
    - 1 small red onion, chopped
    - 1 tablespoon olive oil
    - 1 teaspoon chili powder
    - 1 teaspoon cumin
    - Salt and pepper, to taste
    - 1/2 cup shredded cheddar cheese (optional)
    - 1/4 cup chopped fresh cilantro (optional)
    
    Instructions:
    1. Preheat the oven to 375°F (190°C).
    2. Spread the cooked quinoa in a single layer on a baking sheet.
    3. Roast the quinoa in the oven for 15-20 minutes, or until lightly golden brown and crispy.
    4. In a large skillet over medium heat, add the black beans, red bell pepper, red onion, olive oil, chili powder, cumin, salt, and pepper. Cook for 5-7 minutes, or until the vegetables are tender and slightly browned.
    5. Remove the skillet from the heat and stir in the shredded cheddar cheese (if using) and chopped cilantro (if using).
    6. Serve the black bean and quinoa mixture hot, garnished with additional cilantro if desired.
    
    This meal is high in protein, low in calories, and packed with flavor. It's also a great option for those following Atkins or other low-carb diets. Enjoy!
    


```python
# -- Additional queries demo --
additional_queries = [
    'What is a good low-fat dessert with chocolate?',
    'Suggest a quick vegetarian dinner under 30 minutes',
    'I need a high-protein breakfast recipe',
]

for q in additional_queries:
    print(f'\n{"=" * 80}')
    print(f'QUERY: {q}')
    print(f'{"=" * 80}\n')
    
    results = retrieve(q, model, index, metadata, top_k=5)
    print('Top retrieved recipes:')
    for r in results:
        print(f'  [{r["rank"]}] {r["name"]} (score={r["score"]:.4f})')
    
    print(f'\nGemini Answer:')
    print(f'{"-" * 40}')
    ans = generate_answer(q, results)
    print(ans)
```

    
    ================================================================================
    QUERY: What is a good low-fat dessert with chocolate?
    ================================================================================
    
    Top retrieved recipes:
      [1] oops  there it is   chocolate cake low fat (score=0.6795)
      [2] almost  no fat chocolate cake (score=0.6475)
      [3] warm chocolate cakes (score=0.6457)
      [4] the ultimate chocolate cake (score=0.6434)
      [5] beautiful chocolate disks with seeds and fruits (score=0.6268)
    
    Gemini Answer:
    ----------------------------------------
    

    A good low-fat dessert with chocolate would be a fruit salad. Here are some suggestions:
    
    1. Fresh berries: Choose berries such as strawberries, blueberries, raspberries, and blackberries. You can also add some sliced kiwi or mango for a tropical twist.
    
    2. Fresh fruit: Choose a variety of fresh fruits such as pineapple, mango, papaya, and kiwi. You can also add some sliced kiwi or mango for a tropical twist.
    
    3. Fruit salad with yogurt: Mix fresh fruits with plain yogurt for a creamy and healthy dessert. You can also add some granola or chopped nuts for added crunch.
    
    4. Fruit and nut salad: Combine fresh fruits such as strawberries, blueberries, raspberries, and kiwi with chopped almonds, pecans, and walnuts.
    
    5. Fruit and granola: Combine fresh fruits such as strawberries, blueberries, raspberries, and kiwi with granola and chopped nuts.
    
    6. Fruit and honey: Combine fresh fruits such as strawberries, blueberries, raspberries, and kiwi with honey for a sweet and healthy dessert.
    
    Remember to choose low-fat fruits and nuts to keep your dessert low in fat and calories. Enjoy!
    
    ================================================================================
    QUERY: Suggest a quick vegetarian dinner under 30 minutes
    ================================================================================
    
    Top retrieved recipes:
      [1] vegetarian barbecue sandwiches slow cooker (score=0.6392)
      [2] basil roasted vegetables over couscous (score=0.6269)
      [3] australian ww bacon   vegetable pasta 5 pts (score=0.6252)
      [4] african potato stew (score=0.6160)
      [5] vegetarian parmesan cutlets (score=0.6148)
    
    Gemini Answer:
    ----------------------------------------
    

    Here's a quick and easy vegetarian dinner recipe that's under 30 minutes to prepare:
    
    Ingredients:
    - 1 cup cooked quinoa
    - 1 can black beans, drained and rinsed
    - 1 red bell pepper, diced
    - 1 yellow bell pepper, diced
    - 1 small zucchini, diced
    - 1 small yellow onion, diced
    - 1 tablespoon olive oil
    - 1 teaspoon chili powder
    - 1/2 teaspoon ground cumin
    - 1/4 teaspoon salt
    - Fresh cilantro, chopped (optional)
    
    Instructions:
    1. Preheat oven to 375°F (190°C).
    2. Spread quinoa in a single layer on a baking sheet.
    3. Bake for 15-20 minutes, or until lightly toasted and fragrant.
    4. In a large skillet, heat olive oil over medium heat.
    5. Add diced bell peppers, zucchini, and yellow onion. Cook for 5-7 minutes, or until vegetables are slightly softened.
    6. Add chili powder, cumin, and salt to the skillet. Stir to combine.
    7. Add cooked quinoa to the skillet and stir to combine.
    8. Cook for an additional 2-3 minutes, or until the vegetables are heated through.
    9. Serve
    
    ================================================================================
    QUERY: I need a high-protein breakfast recipe
    ================================================================================
    
    Top retrieved recipes:
      [1] zone style flourless pancakes (score=0.6865)
      [2] bacon   egg sandwiches  breakfast  lunch mmmmmmm (score=0.6661)
      [3] wake up sandwiches (score=0.6608)
      [4] yummy quick   healthy breakfast sandwich (score=0.6598)
      [5] sweet and salty breakfast sandwich (score=0.6466)
    
    Gemini Answer:
    ----------------------------------------
    

    Sure, here's a high-protein breakfast recipe that you can try:
    
    --- Evidence 1 (score: 0.6865) ---
    Recipe: zone style flourless pancakes
    Ingredients: oatmeal, 1% fat cottage cheese, egg, egg white, sugar, vanilla, baking powder, cinnamon
    Calories: 302.8 kcal | Protein: 45.0 %DV | Fat: 10.0 %DV
    Prep Time: 10 minutes
    Average Rating: 5.0/5 (5 reviews)
    Review Summary: I used dry slow-cooking oatmeal (which has more carbs) and 1 cup raspberries (blended into the batter), which upped the carbs quite a bit, and used 2 egg whites.  I also stirred 2 T. slivered almonds into the batter before cooking. To counter the lack of protein, I made 3 scrambled egg whites topped with 2 oz lowfat cheese.  This served DH and myself.  The pancakes were sweet enough that we just ate them as is, with no toppings.  I will be making these pancakes ahead of time & keeping them in the fridge.  I'll definitely be making this again!
    

## 11. Utility: Load Persisted Index

For future use, here is how to reload the saved FAISS index and metadata without re-encoding.


```python
def load_faiss_index(index_path, metadata_path):
    """Load a previously saved FAISS index and its metadata."""
    idx = faiss.read_index(str(index_path))
    with open(metadata_path, 'rb') as f:
        meta = pickle.load(f)
    print(f'Loaded FAISS index with {idx.ntotal:,} vectors.')
    return idx, meta

# Demo reload
reloaded_index, reloaded_meta = load_faiss_index(INDEX_PATH, METADATA_PATH)

# Quick check
test_results = retrieve('simple pasta recipe', model, reloaded_index, reloaded_meta, top_k=3)
for r in test_results:
    print(f'  [{r["rank"]}] {r["name"]} (score={r["score"]:.4f})')

print('\n[OK] Reloaded index works correctly.')
```

    Loaded FAISS index with 20,000 vectors.
    

      [1] artichoke chicken pasta w sun dried tomatoes (score=0.6885)
      [2] basic tortellini pasta (score=0.6859)
      [3] baked spaghetti olivetti (score=0.6732)
    
    [OK] Reloaded index works correctly.
    

## 12. Summary

| Component | Details |
|-----------|--------|
| Embedding Model | `all-MiniLM-L6-v2` (384-dim, SentenceTransformers) |
| Vector Store | FAISS `IndexFlatIP` (exact cosine similarity via normalised inner product) |
| Knowledge Base | 20,000 top-rated recipes (sampled from ~231K for CPU efficiency) |
| Retrieval | Top-k semantic nearest-neighbour search |
| RAG Prompt | Grounded — LLM must answer only from retrieved evidence |
| Persisted Artifacts | `food_kb.index` (FAISS) + `food_kb_meta.pkl` (metadata) |

### Key Design Decisions
- **Controlled generation**: The system prompt explicitly forbids the LLM from using external knowledge.
- **Citation required**: The LLM must cite recipe names, ensuring traceability.
- **Evidence-gated**: If the retrieved evidence does not support the query, the LLM is instructed to say so.

This pipeline can be extended with:
- **Re-ranking** (e.g. cross-encoder) for higher precision.
- **Hybrid search** (BM25 + dense) for better recall.
- **Streaming LLM output** for better user experience.
- **User feedback loop** to continuously improve retrieval quality.
