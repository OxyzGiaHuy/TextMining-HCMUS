import os
import json
import asyncio
import numpy as np
import httpx
import argparse
from datetime import datetime
from pyvi import ViTokenizer
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from metrics import compute_f1, compute_bleu, compute_bertscore, llm_metrics_trio, llm_recall_judge

# --- CONFIG ---
DATA_FILE = "/data2/Medical/ColonMoE/TextMining/ViHERMES/dataset/dataset.jsonl"
OLLAMA_URL = "http://localhost:11434/v1"
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "qwen2.5:latest"
RAG_PROMPT_TEMPLATE = """Dựa vào các thông tin y khoa sau đây:
{context}

Hãy trả lời câu hỏi: {question}"""

# Global Reranker (Shared across all methods)
reranker = CrossEncoder('BAAI/bge-reranker-v2-m3', device='cuda', trust_remote_code=True)

# --- UTILS ---
async def get_embeddings(texts, client):
    tasks = [client.post(f"{OLLAMA_URL}/embeddings", json={"model": EMBED_MODEL, "input": t}) for t in texts]
    resps = await asyncio.gather(*tasks)
    return np.array([r.json()["data"][0]["embedding"] for r in resps])

async def generate_response(prompt, client):
    resp = await client.post(f"{OLLAMA_URL}/chat/completions", json={
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1
    })
    return resp.json()["choices"][0]["message"]["content"]

def rrf(rankings, k=60):
    scores = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking):
            scores[doc] = scores.get(doc, 0) + 1 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

# --- CORE PIPELINE ---
async def run_unified_eval(method, limit):
    # Create timestamped experiment directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join("exp", timestamp)
    os.makedirs(exp_dir, exist_ok=True)
    
    checkpoint_file = os.path.join(exp_dir, f"checkpoint_{method}.json")
    results_cache = []
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, "r") as f:
                results_cache = json.load(f)
        except: pass
    processed_ids = {x["id"] for x in results_cache}

    async with httpx.AsyncClient(timeout=300.0) as client:
        # 1. Load Data
        print(f"Loading data from {DATA_FILE}...")
        records = []
        with open(DATA_FILE, "r") as f:
            for line in f: records.append(json.loads(line))
        
        unique_blocks = []
        for r in records:
            ev = r.get("evidence", {})
            if isinstance(ev, dict):
                block = " ".join([str(v) for v in ev.values()])
                if block and block not in unique_blocks: unique_blocks.append(block)

        # 2. Preparation (Embedding/Indexing)
        bm25 = None
        chunk_embeddings = None
        
        if method in ["bm25", "hybrid"]:
            print(f"Indexing {len(unique_blocks)} blocks for BM25...")
            tokenized_corpus = [ViTokenizer.tokenize(doc).split() for doc in unique_blocks]
            bm25 = BM25Okapi(tokenized_corpus)
            
        if method in ["dense", "hybrid"]:
            print(f"Embedding {len(unique_blocks)} unique context blocks (Dense Strategy)...")
            eb_batch_size = 16
            all_embeddings = []
            for i in range(0, len(unique_blocks), eb_batch_size):
                batch = unique_blocks[i:i+eb_batch_size]
                embs = await get_embeddings(batch, client)
                all_embeddings.extend(embs)
                if i % 100 == 0: print(f" -> Embedded {len(all_embeddings)}/{len(unique_blocks)}")
            chunk_embeddings = np.array(all_embeddings)

        # 3. Evaluation Loop
        semaphore = asyncio.Semaphore(10)
        async def eval_one(idx, r):
            async with semaphore:
                q = r["question"]
                gt = r["answer"]
                
                # A. Retrieval
                retrieved_hits = []
                if method == "dense":
                    q_emb_resp = await get_embeddings([q], client)
                    q_v_scores = np.dot(chunk_embeddings, q_emb_resp[0])
                    dense_idx = np.argsort(q_v_scores)[-20:][::-1]
                    retrieved_hits = [unique_blocks[i] for i in dense_idx]
                    
                elif method == "bm25":
                    q_tokens = ViTokenizer.tokenize(q).split()
                    retrieved_hits = bm25.get_top_n(q_tokens, unique_blocks, n=20)
                    
                elif method == "hybrid":
                    # BM25 Top-50
                    q_tokens = ViTokenizer.tokenize(q).split()
                    bm25_hits = bm25.get_top_n(q_tokens, unique_blocks, n=50)
                    # Dense Top-50
                    q_emb_resp = await get_embeddings([q], client)
                    q_v_scores = np.dot(chunk_embeddings, q_emb_resp[0])
                    dense_idx = np.argsort(q_v_scores)[-50:][::-1]
                    dense_hits = [unique_blocks[i] for i in dense_idx]
                    # RRF Top-20
                    hybrid_ranked = rrf([bm25_hits, dense_hits])
                    retrieved_hits = [d for d, s in hybrid_ranked[:20]]

                # B. Reranking (Common for all)
                pairs = [[q, doc] for doc in retrieved_hits]
                rr_scores = reranker.predict(pairs)
                ranked = sorted(zip(retrieved_hits, rr_scores), key=lambda x: x[1], reverse=True)
                top_5 = [d for d, _ in ranked[:5]]
                context = "\n".join(top_5)
                
                # C. Generation
                prompt = RAG_PROMPT_TEMPLATE.format(context=context, question=q)
                pred = await generate_response(prompt, client)
                
                # D. Metrics
                f1 = compute_f1(pred, gt)
                bleu = compute_bleu(pred, gt)
                j, f, rel = await llm_metrics_trio(q, pred, gt, context)
                rec = await llm_recall_judge(context, gt)
                
                res = {
                    "id": idx, "question": q, "f1": f1, "bleu": bleu, "judge": j, 
                    "faithfulness": f, "relevancy": rel, "recall": rec,
                    "pred": pred, "truth": gt
                }
                nonlocal results_cache
                results_cache.append(res)
                with open(checkpoint_file, "w") as f_out: json.dump(results_cache, f_out)
                print(f"[{method.upper()}] Sample {idx+1}/{len(records)} | F1: {f1:.2f} | J: {j} | Rec: {rec}")
                return res

        to_process = [(i, r) for i, r in enumerate(records) if i not in processed_ids]
        if limit: to_process = to_process[:limit]
        
        if to_process:
            print(f"Starting execution for {len(to_process)} samples (Method: {method})...")
            tasks = [eval_one(idx, r) for idx, r in to_process]
            await asyncio.gather(*tasks)

        # E. Final BERTScore Calculation
        if results_cache:
            valid_for_bert = [x for x in results_cache if "pred" in x and "truth" in x and "bertscore" not in x]
            if valid_for_bert:
                print(f"Calculating BERTScore for {len(valid_for_bert)} new samples...")
                preds = [x["pred"] for x in valid_for_bert]
                truths = [x["truth"] for x in valid_for_bert]
                scores = compute_bertscore(preds, truths)
                for item, score in zip(valid_for_bert, scores):
                    item["bertscore"] = score
                    
            # Save final results with all metrics
            with open(checkpoint_file, "w") as f_out: json.dump(results_cache, f_out)
            
            avg_f1 = np.mean([x["f1"] for x in results_cache])
            avg_bleu = np.mean([x.get("bleu", 0) for x in results_cache])
            avg_bert = np.mean([x.get("bertscore", 0) for x in results_cache])
            avg_j = np.mean([x["judge"] for x in results_cache])
            avg_rec = np.mean([x["recall"] for x in results_cache])
            
            print(f"\n>>>> FINAL RESULTS FOR {method.upper()} ({len(results_cache)} samples):")
            print(f"Avg F1: {avg_f1:.4f} | BLEU: {avg_bleu:.4f} | BERTScore: {avg_bert:.4f} | J: {avg_j*100:.2f}% | Rec: {avg_rec*100:.2f}%")
            
            final_results_file = os.path.join(exp_dir, f"results_{method}.json")
            with open(final_results_file, "w") as f:
                json.dump({
                    "method": method, "F1": avg_f1, "BLEU": avg_bleu, "BERTScore": avg_bert,
                    "LLM_Judge": avg_j, "Recall@5": avg_rec, "Total": len(results_cache),
                    "timestamp": timestamp
                }, f)
            print(f"Results saved to {final_results_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, required=True, choices=["dense", "bm25", "hybrid"])
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    asyncio.run(run_unified_eval(args.method, args.limit))
