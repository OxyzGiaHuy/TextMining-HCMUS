# 🧬 Recommended Model Configurations for Medical RAG Experiments

This document lists the recommended Embedding and Reranker models for benchmarking on the ViHERMES/ColonMoE dataset.

## 📥 Embedding Models (Bi-Encoders)
| Name | HuggingFace Model ID | Best For |
| :--- | :--- | :--- |
| **BGE-M3** | `bge-m3` (Ollama) | Multi-task, Multilingual (Modern SOTA). |
| **Nomic Embed** | `nomic-embed-text` (Ollama) | (Current Baseline) Long context (8k tokens). |
| **mE5-Large** | `mxbai-embed-large` or `mE5` (Ollama) | High precision. |

> [!TIP]
> To use these models with the existing Ollama structure, run:
> - `ollama pull bge-m3`
> - `ollama pull nomic-embed-text`
> - `ollama pull mxbai-embed-large`

## 🔍 Reranker Models (Cross-Encoders)
| Name | HuggingFace Model ID | Best For |
| :--- | :--- | :--- |
| **BGE-v2-m3** | `BAAI/bge-reranker-v2-m3` | (Current Baseline) Multilingual, high accuracy. |
| **PhoRanker** | `itdainb/PhoRanker` | Specializing in Vietnamese queries. |
| **ColBERT** | `answerdotai/answerai-colbert-small-v1` | Fast reranking of large candidates. |
| **BGE-Large** | `BAAI/bge-reranker-large` | Standard large multilingual reranker. |

## 🛠 Integration Guide

To test a specific model, update the following variables in `eval_baselines.py`:

### For Reranker:
```python
# Change the model path in CrossEncoder initialization:
reranker = CrossEncoder('it-vnu-hcm/vietnamese-reranker', device='cuda')
```

### For Embedding:
If using `SentenceTransformers` instead of Ollama for more control over HF models:
```python
from sentence_transformers import SentenceTransformer
embed_model = SentenceTransformer('BAAI/bge-m3', device='cuda')
# Update get_embeddings() function to use this model
```

---
*Created for HCMUS TextMining & ColonMoE Research.*
