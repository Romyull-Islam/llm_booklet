"""
word_quality.py: section 1.7, the claim that the word-level model scores
"100.0%, over five seeds, and 100.0% at every generation length from 100 to 800".

Loads ckpt_words.pt, generates with <unk> banned, and scores generated text only.
"""
import re, sys, torch
from model import Transformer
from word_tok import WordTokenizer, WORD_RE
from sample_quality import vocab_from_corpus, score_sample

ck = torch.load(sys.argv[1] if len(sys.argv) > 1 else "ckpt_words.pt",
                map_location="cpu", weights_only=False)
model = Transformer(ck["cfg"]); model.load_state_dict(ck["model"]); model.eval()

itos = ck["itos"]
stoi = {t: i for i, t in enumerate(itos)}


def encode(s):
    return [stoi.get(t, 0) for t in WORD_RE.findall(s)]


def decode(ids):
    out = []
    for i in ids:
        t = itos[int(i)]
        if t == "\n":
            out.append("\n")
        elif re.match(r"[A-Za-z0-9']", t):
            out.append((" " if out and not out[-1].endswith("\n") else "") + t)
        else:
            out.append(t)
    return "".join(out).strip()


train_vocab = vocab_from_corpus("input.txt")
prompt_ids = encode("KING RICHARD")
prompt = torch.tensor([prompt_ids])

print(f"vocab {len(itos)} | <unk> banned at index 0")
print(f"{'tokens':>7} {'real-word rate':>15}   per-seed (5 seeds)")
for n in (100, 128, 200, 400, 800):
    rates, unk_hits = [], 0
    for seed in range(5):
        torch.manual_seed(seed)
        out = model.generate(prompt, n, temperature=0.8, top_k=40, ban_tokens=[0])
        gen_ids = out[0].tolist()[len(prompt_ids):]
        unk_hits += sum(1 for i in gen_ids if int(i) == 0)
        rates.append(score_sample(decode(gen_ids), train_vocab)["real_rate"])
    mean = sum(rates) / len(rates)
    print(f"{n:>7} {mean:>14.1%}   " + " ".join(f"{r:.1%}" for r in rates)
          + (f"   <unk> emitted: {unk_hits}" if unk_hits else ""))

print("\n--- sample, seed 0 ---")
torch.manual_seed(0)
out = model.generate(prompt, 120, temperature=0.8, top_k=40, ban_tokens=[0])
print(decode(out[0].tolist()))
print("\n--- sample, seed 3 ---")
torch.manual_seed(3)
out = model.generate(prompt, 120, temperature=0.8, top_k=40, ban_tokens=[0])
print(decode(out[0].tolist()))
