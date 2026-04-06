# TextMining-HCMUS: Advanced Medical RAG for ColonMoE

## 🩺 Overview
This repository contains the evaluation and development pipeline for **ColonMoE**, a specialized Medical RAG (Retrieval-Augmented Generation) system focused on colon cancer research and clinical reasoning. The project leverages the **ViHERMES** dataset to benchmark and optimize various retrieval and generation strategies for Vietnamese medical contexts.

## 📊 Benchmark Leaderboard
The following table provides a detailed comparison of different RAG architectures and their backbone configurations on the ViHERMES dataset (1,561 samples).

| Method | Strategy | LLM | Embedding | Reranker | F1 | BLEU | BERTScore | Judge | Recall@5 |
| :--- | :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **PureRAG** | BM25 | Qwen2.5-7B | - | BGE-v2-m3 | 0.4893 | 0.2189 | 0.7815 | 89.88% | 76.81% |
| **PureRAG** | Dense | Qwen2.5-7B | Nomic-Embed | BGE-v2-m3 | 0.4170 | 0.1613 | 0.7519 | 70.28% | 48.81% |
| **PureRAG** | Hybrid | Qwen2.5-7B | Nomic-Embed | BGE-v2-m3 | **0.4888** | **0.2192** | **0.7816** | **90.58%** | **75.34%** |
| **LightRAG** | Local | (Queued) | - | - | - | - | - | - | - |
| **LightRAG** | Global | (Queued) | - | - | - | - | - | - | - |

> [!NOTE]
> - **LLM**: `Qwen2.5-7B-Instruct`
> - **Embedding**: `nomic-embed-text`
> - **Reranker**: `BAAI/bge-reranker-v2-m3`
> - **Hybrid**: BM25 + Dense combined via RRF.

## 🚀 Development Roadmap

### 1. PureRAG Execution (Baseline Phase)
Establishing strong baselines using standard retrieval techniques.
- **BM25**: Classic term-based retrieval.
- **Dense Retrieval**: Semantic search using `nomic-embed-text`.
- **Hybrid Search**: Combining BM25 and Dense scores.
- **Reranking**: Refined using a Cross-Encoder for high-precision selection.

### 2. Transition to LightRAG (Advanced Phase)
Integrating **LightRAG**, a graph-enhanced RAG framework, to provide:
- **Graph-Based Indexing**: Capturing complex relationships between medical entities.
- **Multi-Hop Reasoning**: Better clinical reasoning for complex queries.
- **Holistic Summarization**: Global knowledge synthesis.

## 📝 To-Do List (PureRAG Optimization)
### [ ] LLM Model Experiments
- [ ] Test `Qwen2.5-7B-Instruct` (Current baseline)
- [ ] Test `Llama-3.1-8B-Instruct`.
- [ ] Test `Vinallama` / Vietnamese-specialized models.

### [ ] Embedding Model Experiments
- [ ] Test `nomic-embed-text` (Current baseline)
- [ ] Test `keepitreal/vietnamese-sbert`.
- [ ] Test `BAAI/bge-m3`.

### [ ] Reranker Experiments
- [ ] Test `BAAI/bge-reranker-v2-m3` (Current baseline)
- [ ] Test `ColBERT`.
- [ ] No reranking baseline.

## 📐 Evaluation Metrics
- **Correctness (Judge)**: Medical fact accuracy.
- **Faithfulness**: Context-based grounding.
- **Relevancy**: Query alignment.
- **Recall@5**: Retrieval coverage.

## 🛠 Project Structure
- `PureRAG/`: Baseline evaluation scripts.
- `PureRAG/exp/`: Automatically archived experiment results.
- `LightRAG/`: (Ongoing) Graph-based RAG integration.
- `document/`: Centralized results tracking.

---
*Developed by OxyzGiaHuy as part of the HCMUS TextMining research initiative.*