# 15. Claim-Level Faithfulness Check Demo
**Project:** GroundedNutriRec
**Task:** Interactive verification of a RAG-generated explanation against retrieved recipe evidence.

This notebook demonstrates the end-to-end RAG and claim verification pipeline on a single query.


```python
import os
import sys
from pathlib import Path
from sentence_transformers import SentenceTransformer

# Add src to path
sys.path.append(os.path.abspath('../SRC'))

from rag.vector_search import VectorSearchRAG
from rag.generator import GroundedExplanationGenerator
from rag.verifier import FaithfulnessVerifier

# Setup paths
INDEX_PATH = Path('../NOTEBOOKS/dataset/archive_3/faiss_index/food_kb.index')
METADATA_PATH = Path('../NOTEBOOKS/dataset/archive_3/faiss_index/food_kb_meta.pkl')

print('Loading SentenceTransformer...')
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

print('Loading retrieval and generation modules...')
rag_engine = VectorSearchRAG(INDEX_PATH, METADATA_PATH, embed_model)
generator = GroundedExplanationGenerator()
verifier = FaithfulnessVerifier()
print('Setup complete!')
```

    Loading SentenceTransformer...
    

    Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
    


    Loading weights:   0%|          | 0/103 [00:00<?, ?it/s]


    Loading retrieval and generation modules...
    


    Loading weights:   0%|          | 0/201 [00:00<?, ?it/s]


    [transformers] Passing `generation_config` together with generation-related arguments=({'temperature', 'do_sample', 'max_new_tokens'}) is deprecated and will be removed in future versions. Please pass either a `generation_config` object OR all generation parameters explicitly, but not both.
    

    [transformers] `torch_dtype` is deprecated! Use `dtype` instead!
    


    Loading weights:   0%|          | 0/290 [00:00<?, ?it/s]


    Setup complete!
    


```python
# 1. Define query and retrieve evidence
query = 'high protein and fast to prepare'
retrieved = rag_engine.retrieve(query, top_k=1)
doc = retrieved[0]['document']
recipe_name = retrieved[0]['name']

print(f'Query: {query}')
print(f'Recommended Recipe: {recipe_name}')
print(f'Evidence Document:\n{doc}')
```

    Query: high protein and fast to prepare
    Recommended Recipe: three bean salad  high protein
    Evidence Document:
    Recipe: three bean salad  high protein
    Ingredients: cut green beans, red kidney beans, chickpeas, onion, fat-free italian salad dressing
    Calories: 213.3 kcal | Protein: 21.0 %DV | Fat: 2.0 %DV
    Prep Time: 185 minutes
    Average Rating: 5.0/5 (1 reviews)
    Review Summary: Iâ€™m on a high protein diet therefore I am trying all sort of high protein recipes.  I tried this recipe and it came out delicious.  I can save it in the fridge and have it ready with grilled chicken or any type of grilled meat.
    


```python
# 2. Generate grounded explanation
from rag.generator import RAG_SYSTEM_PROMPT

evidence_block = f"--- Evidence 1 ---\n{doc}"
prompt = (
    f'<|system|>\n'
    f'{RAG_SYSTEM_PROMPT}\n'
    f'=== EVIDENCE ===\n'
    f'{evidence_block}</s>\n'
    f'<|user|>\n'
    f'{query}</s>\n'
    f'<|assistant|>\n'
)

explanation = generator.generate(prompt)
print(f'Generated Explanation:\n{explanation}')
```

    [transformers] Ignoring clean_up_tokenization_spaces=True for BPE tokenizer LlamaTokenizer. The clean_up_tokenization post-processing step is designed for WordPiece tokenizers and is destructive for BPE (it strips spaces before punctuation). Set clean_up_tokenization_spaces=False to suppress this warning, or set clean_up_tokenization_spaces_for_bpe_even_though_it_will_corrupt_output=True to force cleanup anyway.
    

    Generated Explanation:
    Sure, here's an updated answer based on the given evidence:
    
    Recipe: three bean salad with high protein
    Ingredients: cut green beans, red kidney beans, chickpeas, onion, fat-free italian salad dressing
    Calories: 213.3 kcal | Protein: 21.0 %DV | Fat: 2.0 %DV
    Prep Time: 185 minutes
    Average Rating: 5.0/5 (1 reviews)
    Review Summary: I'm on a high protein diet and I've been trying out different recipes. This one is a winner! It's quick to prepare and delicious. I can save it in the fridge and have it ready with grilled chicken or any type of grilled meat.
    


```python
# 3. Extract and verify claims
import re
def split_claims(text):
    sentences = re.split(r'[.!?]\s*', text)
    claims = []
    for sent in sentences:
        sent = sent.strip()
        if not sent: continue
        clauses = re.split(r'\b(and|but|additionally|furthermore)\b', sent, flags=re.IGNORECASE)
        for c in clauses:
            c = c.strip()
            if len(c.split()) > 3 and c.lower() not in ['and', 'but', 'additionally', 'furthermore']:
                claims.append(c)
    return claims

claims = split_claims(explanation)
print(f'Extracted {len(claims)} atomic claims:')
for idx, c in enumerate(claims):
    print(f'  [{idx}] {c}')

print('\nVerifying claims against evidence...')
verdicts = verifier.verify_batch(claims, [doc]*len(claims))

print('\n=== Verification Results ===')
for c, v in zip(claims, verdicts):
    print(f'Claim: "{c}" -> Verdict: {v}')
```

    Extracted 10 atomic claims:
      [0] Sure, here's an updated answer based on the given evidence:
    
    Recipe: three bean salad with high protein
    Ingredients: cut green beans, red kidney beans, chickpeas, onion, fat-free italian salad dressing
    Calories: 213
      [1] 3 kcal | Protein: 21
      [2] 0 %DV | Fat: 2
      [3] 0 %DV
    Prep Time: 185 minutes
    Average Rating: 5
      [4] 0/5 (1 reviews)
    Review Summary: I'm on a high protein diet
      [5] I've been trying out different recipes
      [6] This one is a winner
      [7] It's quick to prepare
      [8] I can save it in the fridge
      [9] have it ready with grilled chicken or any type of grilled meat
    
    Verifying claims against evidence...
    

    
    === Verification Results ===
    Claim: "Sure, here's an updated answer based on the given evidence:
    
    Recipe: three bean salad with high protein
    Ingredients: cut green beans, red kidney beans, chickpeas, onion, fat-free italian salad dressing
    Calories: 213" -> Verdict: Supported
    Claim: "3 kcal | Protein: 21" -> Verdict: Unsupported
    Claim: "0 %DV | Fat: 2" -> Verdict: Unsupported
    Claim: "0 %DV
    Prep Time: 185 minutes
    Average Rating: 5" -> Verdict: Unsupported
    Claim: "0/5 (1 reviews)
    Review Summary: I'm on a high protein diet" -> Verdict: Unsupported
    Claim: "I've been trying out different recipes" -> Verdict: Supported
    Claim: "This one is a winner" -> Verdict: Supported
    Claim: "It's quick to prepare" -> Verdict: Supported
    Claim: "I can save it in the fridge" -> Verdict: Unsupported
    Claim: "have it ready with grilled chicken or any type of grilled meat" -> Verdict: Unsupported
    
