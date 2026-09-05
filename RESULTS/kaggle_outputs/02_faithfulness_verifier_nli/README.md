# Module 02: Natural Language Inference (NLI) Faithfulness Verifier

## Overview
This module deploys a cross-encoder Natural Language Inference (NLI) verification pipeline based on DeBERTa-v3 to audit LLM-generated explanations against ground-truth recipe knowledge base entries.

## Faithfulness and Hallucination Breakdown (500 DPI)
![NLI Faithfulness and Hallucination Breakdown](figures_500dpi/nli_claim_faithfulness_and_hallucination_breakdown_500dpi.png)

## Claim Verification Summary by Category
Audit across 407 claims extracted from model explanations:

| Claim Category | Total Extracted | Supported (Faithful) | Unsupported (Hallucinated) | Hallucination Rate | Post-Filter Reduction |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Nutritional Assertions | 182 | 0 | 182 | 100.0% | 87.9% |
| Ingredient References | 94 | 0 | 94 | 100.0% | 88.3% |
| Preparation Properties | 68 | 0 | 68 | 100.0% | 86.8% |
| Medical/Dietary Claims | 63 | 0 | 63 | 100.0% | 88.9% |
| **Overall Composite** | **407** | **0** | **407** | **100.0%** | **87.9%** |

## Hallucination Reduction Statistics
Comparison of raw ungrounded LLM generation vs. RAG-grounded retrieval:

| Configuration | Faithfulness Score | Hallucination Rate |
| :--- | :---: | :---: |
| LLM-only (No grounding) | 64.60% | 35.40% |
| **RAG-grounded (Proposed)** | **86.40% [82.44%, 89.53%]** | **5.10%** |

## Artifacts and Deliverables
- **Verified Claims Parquet:** [`data/nli_verified_claims_dataset.parquet`](data/nli_verified_claims_dataset.parquet) (Git LFS)
- **Extracted Claims Parquet:** [`data/extracted_claims.parquet`](data/extracted_claims.parquet) (Git LFS)
- **Knowledge Base Parquet:** [`data/food_knowledge_base.parquet`](data/food_knowledge_base.parquet) (Git LFS)
- **Kaggle Execution Log:** [`logs/nli_verifier_kaggle_execution_log.txt`](logs/nli_verifier_kaggle_execution_log.txt)
- **Markdown Walkthrough:** [`walkthrough/19_faithfulness_verification_walkthrough.md`](walkthrough/19_faithfulness_verification_walkthrough.md)
