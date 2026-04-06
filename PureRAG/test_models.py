import torch
import argparse
from sentence_transformers import SentenceTransformer, CrossEncoder

def test_models(embed_id, rerank_id):
    print(f"--- Testing Models ---")
    print(f"Embedding ID: {embed_id}")
    print(f"Reranker ID: {rerank_id}")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 1. Test Embedding
    print(f"\n[1/2] Loading Embedding Model...")
    try:
        model = SentenceTransformer(embed_id, device=device, trust_remote_code=True)
        sentences = ["Ung thư đại tràng là gì?", "Cách phòng ngừa bệnh ung thư."]
        embeddings = model.encode(sentences)
        print(f"✅ Embedding Success! Output shape: {embeddings.shape}")
    except Exception as e:
        print(f"❌ Embedding Failed: {e}")

    # 2. Test Reranker
    print(f"\n[2/2] Loading Reranker Model...")
    try:
        reranker = CrossEncoder(rerank_id, device=device, trust_remote_code=True)
        query = "Triệu chứng ung thư?"
        docs = [
            "Triệu chứng bao gồm đau bụng và thay đổi thói quen đại tiện.",
            "Trái cây rất tốt cho sức khỏe."
        ]
        scores = reranker.predict([[query, d] for d in docs])
        print(f"✅ Reranker Success! Scores: {scores}")
    except Exception as e:
        print(f"❌ Reranker Failed: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--embed_id", type=str, default="BAAI/bge-m3")
    parser.add_argument("--rerank_id", type=str, default="it-vnu-hcm/vietnamese-reranker")
    args = parser.parse_args()
    
    test_models(args.embed_id, args.rerank_id)
