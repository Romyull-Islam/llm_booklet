"""Objective quality metric for a tiny LM: what fraction of generated words are real?

Loss tells you the model is improving. It does not tell you whether the output is
readable. This does, and it takes two seconds to compute.
"""
import re, sys, torch
from model import Config, Transformer
from train import CharTokenizer

def vocab_from_corpus(path):
    return set(re.findall(r"[a-z']+", open(path, encoding="utf-8").read().lower()))

def score_sample(text, train_vocab):
    words = re.findall(r"[A-Za-z']+", text)
    if not words:
        return {"words": 0, "real_rate": 0.0, "invented": []}
    invented = [w for w in words if w.lower() not in train_vocab]
    return {"words": len(words),
            "real_rate": round(1 - len(invented) / len(words), 4),
            "invented": invented}

if __name__ == "__main__":
    ckpt_path = sys.argv[1] if len(sys.argv) > 1 else "ckpt_dense.pt"
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = Transformer(ck["cfg"]); model.load_state_dict(ck["model"]); model.eval()
    tok = CharTokenizer("")
    tok.chars = ck["chars"]
    tok.stoi = {c: i for i, c in enumerate(tok.chars)}
    tok.itos = {i: c for c, i in tok.stoi.items()}

    train_vocab = vocab_from_corpus("input.txt")
    torch.manual_seed(0)
    prompt = torch.tensor([tok.encode("KING RICHARD")])
    out = model.generate(prompt, 400, temperature=0.8, top_k=40)
    text = tok.decode(out[0].tolist())
    r = score_sample(text, train_vocab)
    print(text)
    print("\n" + "=" * 50)
    print(f"words {r['words']} | real-word rate {r['real_rate']:.1%}")
    print(f"invented: {r['invented']}")
