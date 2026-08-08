"""
quality_vs_length.py: section 1.7, "Quality falls off a cliff past the trained context."

Three seeds per row, scoring ONLY the generated text (the prompt is stripped),
on a model trained at max_seq_len=128.
"""
import sys, torch
from model import Transformer
from train import CharTokenizer
from sample_quality import vocab_from_corpus, score_sample

ck = torch.load(sys.argv[1] if len(sys.argv) > 1 else "ckpt_dense.pt",
                map_location="cpu", weights_only=False)
model = Transformer(ck["cfg"]); model.load_state_dict(ck["model"]); model.eval()
tok = CharTokenizer("")
tok.chars = ck["chars"]
tok.stoi = {c: i for i, c in enumerate(tok.chars)}
tok.itos = {i: c for c, i in tok.stoi.items()}

train_vocab = vocab_from_corpus("input.txt")
PROMPT = "KING RICHARD"
prompt_ids = tok.encode(PROMPT)
prompt = torch.tensor([prompt_ids])

print(f"trained context: max_seq_len={ck['cfg'].max_seq_len}")
print(f"{'tokens':>7} {'real-word rate':>15}   seeds")
for n in (100, 128, 200, 400, 800):
    rates = []
    for seed in (0, 1, 2):
        torch.manual_seed(seed)
        out = model.generate(prompt, n, temperature=0.8, top_k=40)
        gen = tok.decode(out[0].tolist()[len(prompt_ids):])   # generated text only
        rates.append(score_sample(gen, train_vocab)["real_rate"])
    mean = sum(rates) / len(rates)
    print(f"{n:>7} {mean:>14.1%}   " + " ".join(f"{r:.1%}" for r in rates))
