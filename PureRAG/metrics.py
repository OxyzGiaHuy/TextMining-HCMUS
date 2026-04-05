import os
import re
import json
import asyncio
import numpy as np
import httpx
from pyvi import ViTokenizer
from collections import Counter
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from bert_score import score as bert_score_func

# --- CONFIG ---
OLLAMA_URL = "http://localhost:11434/v1"

# --- NLP METRICS ---

def compute_f1(pred: str, truth: str, remove_stopwords: bool = False):
    if not pred or not truth: return 0
    stopwords = {"là", "có", "trong", "để", "cho", "với", "tại", "này", "cũng", "và", "của", "các", "nhưng", "rồi", "mà"}
    
    pred_seg = ViTokenizer.tokenize(pred.strip().lower())
    truth_seg = ViTokenizer.tokenize(truth.strip().lower())
    
    pred_tokens = pred_seg.split()
    truth_tokens = truth_seg.split()
    
    if remove_stopwords:
        pred_tokens = [w for w in pred_tokens if w not in stopwords]
        truth_tokens = [w for w in truth_tokens if w not in stopwords]
        
    if not pred_tokens or not truth_tokens: return 0
    
    common = Counter(pred_tokens) & Counter(truth_tokens)
    num_same = sum(common.values())
    if num_same == 0: return 0
    
    precision = 1.0 * num_same / len(pred_tokens)
    recall = 1.0 * num_same / len(truth_tokens)
    return (2 * precision * recall) / (precision + recall)

def compute_bleu(pred: str, truth: str):
    if not pred or not truth: return 0
    pred_tokens = ViTokenizer.tokenize(pred.strip().lower()).split()
    truth_tokens = [ViTokenizer.tokenize(truth.strip().lower()).split()]
    
    chencherry = SmoothingFunction()
    # Using method1 for smoothing
    score = sentence_bleu(truth_tokens, pred_tokens, smoothing_function=chencherry.method1)
    return score

def compute_bertscore(preds: list, truths: list, lang="vi"):
    if not preds or not truths:
        return []
    # bert_score.score returns (P, R, F1)
    # We use 'bert-base-multilingual-cased' for Vietnamese stability if not specified
    P, R, F1 = bert_score_func(preds, truths, lang=lang, verbose=False, model_type="bert-base-multilingual-cased")
    return F1.numpy().tolist()

# --- LLM JUDGE METRICS ---

async def vllm_judge_call(prompt: str, model="qwen2.5:latest") -> str:
    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            resp = await client.post(f"{OLLAMA_URL}/chat/completions", json={
                "model": model, 
                "messages": [
                    {"role": "system", "content": "Bạn là một vị Giám khảo Y khoa chuyên nghiệp (Medical NLP Expert)."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.0
            })
            return resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            return f"Error: {str(e)}"

async def llm_metrics_trio(q, pred, truth, ctx):
    prompt = f"""Hãy chấm điểm câu trả lời RAG dựa trên các tiêu chí sau:
1. Correctness (Chính xác): Câu trả lời có khớp với Đáp án chuẩn về mặt chuyên môn y tế không?
2. Faithfulness (Trung thực): Câu trả lời có hoàn toàn dựa trên Ngữ cảnh được cung cấp không (không suy diễn ngoài lề)?
3. Relevancy (Liên quan): Câu trả lời có giải quyết trực tiếp câu hỏi không?

Câu hỏi: {q}
Ngữ cảnh: {ctx}
Đáp án chuẩn: {truth}
Câu trả lời của hệ thống: {pred}

Hãy trả về CHỈ 3 số 0 hoặc 1, cách nhau bởi dấu phẩy (Correctness, Faithfulness, Relevancy).
Ví dụ: 1,1,0
Kết quả:"""
    try:
        ans = await vllm_judge_call(prompt)
        bits = [int(x) for x in re.findall(r'[01]', ans)[:3]]
        return bits if len(bits) == 3 else [0, 0, 0]
    except:
        return [0, 0, 0]

async def llm_recall_judge(top_context: str, gt_context: str) -> int:
    prompt = f"""Kiểm tra xem 'Ngữ cảnh' có chứa đủ thông tin để trả lời được 'Đáp án chuẩn' không.
Trả về 1 nếu Có, 0 nếu Không.

Đáp án chuẩn: {gt_context}
Ngữ cảnh: {top_context}
Kết quả (1 hoặc 0):"""
    try:
        ans = await vllm_judge_call(prompt)
        return 1 if "1" in ans else 0
    except:
        return 0

# Test run if executed as script
if __name__ == "__main__":
    p = "Bệnh sởi có thể lây qua đường hô hấp."
    t = "Sởi lây truyền qua các giọt bắn đường hô hấp của người bệnh."
    print(f"F1: {compute_f1(p, t)}")
    print(f"BLEU: {compute_bleu(p, t)}")
    # BERTScore takes time for model download, test with caution
    # print(f"BERTScore: {compute_bertscore([p], [t])}")
