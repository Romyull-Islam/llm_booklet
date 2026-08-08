"""
gen_timing.py: section 1.6, "Time them both and write the ratio down."

The cached path is linear in generated tokens, the uncached path is quadratic,
so the ratio should widen as you generate more.
"""
import sys, time, torch
from model import Transformer
from train import CharTokenizer

ck = torch.load(sys.argv[1] if len(sys.argv) > 1 else "ckpt_dense.pt",
                map_location="cpu", weights_only=False)
model = Transformer(ck["cfg"]); model.load_state_dict(ck["model"]); model.eval()
tok = CharTokenizer("")
tok.chars = ck["chars"]
tok.stoi = {c: i for i, c in enumerate(tok.chars)}
tok.itos = {i: c for c, i in tok.stoi.items()}

prompt = torch.tensor([tok.encode("KING RICHARD")])

def table(label):
    print(f"\n{label}  (max_seq_len={model.cfg.max_seq_len})")
    print(f"{'tokens':>7} {'uncached (s)':>13} {'cached (s)':>11} {'speedup':>8}")
    for n in (100, 200, 400, 800):
        times = {}
        for use_cache in (False, True):
            torch.manual_seed(0)
            model.generate(prompt, 8, temperature=0.8, top_k=40, use_cache=use_cache)  # warm up
            torch.manual_seed(0)
            t0 = time.perf_counter()
            model.generate(prompt, n, temperature=0.8, top_k=40, use_cache=use_cache)
            times[use_cache] = time.perf_counter() - t0
        print(f"{n:>7} {times[False]:>13.3f} {times[True]:>11.3f} {times[False]/times[True]:>7.2f}x")


# As shipped. generate(use_cache=False) slices `idx[:, -max_seq_len:]`, so the
# uncached path is capped at 128 positions per forward: linear, not quadratic.
table("A. uncached path window-capped, as the booklet ships it")

# Remove the cap so the uncached path really does re-run the whole prefix every
# step. This is the comparison the booklet's "quadratic vs linear" claim assumes.
model.cfg.max_seq_len = 4096          # RoPE table grows on demand
table("B. uncached path uncapped, re-running the full prefix each step")
