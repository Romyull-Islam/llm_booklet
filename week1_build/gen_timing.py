"""
gen_timing.py: section 1.6, "Time them both and write the ratio down."

Correctness first. A speedup on wrong output is worthless, so this asserts the
cached and uncached paths produce identical tokens at 64, 128 and 300 generated
tokens -- the last one deliberately past the trained context -- before it reports
a single timing.

The uncached path must re-run the whole prefix each step. An earlier version of
model.py sliced it to max_seq_len, which capped the work per step and made the
naive path linear; the cache then lost on measurement, and past max_seq_len the
two paths diverged outright. That cap is gone.
"""
import argparse
import sys
import time

import torch

ap = argparse.ArgumentParser()
ap.add_argument("ckpt", nargs="?", default="ckpt_dense.pt")
ap.add_argument("--threads", type=int, default=1,
                help="CPU threads. Default 1 to match the booklet's 'one CPU core' table. "
                     "On many cores the quadratic recompute parallelises and the cache "
                     "looks far less impressive than it is per-core.")
args = ap.parse_args()
torch.set_num_threads(args.threads)

from model import Transformer
from train import CharTokenizer

ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
model = Transformer(ck["cfg"])
model.load_state_dict(ck["model"])
model.eval()

tok = CharTokenizer("")
tok.chars = ck["chars"]
tok.stoi = {c: i for i, c in enumerate(tok.chars)}
tok.itos = {i: c for c, i in tok.stoi.items()}

prompt = torch.tensor([tok.encode("KING RICHARD")])


def generate(n, use_cache, seed=0):
    torch.manual_seed(seed)
    # temperature=0 is greedy, so both paths are deterministic and directly comparable.
    return model.generate(prompt, n, temperature=0.0, use_cache=use_cache)


# ---------------------------------------------------------------- correctness
print(f"trained context: max_seq_len={model.cfg.max_seq_len} | threads {args.threads}")
print("\nequivalence check (greedy, so the two paths must agree exactly):")
for n in (64, 128, 300):
    a = generate(n, use_cache=False)
    b = generate(n, use_cache=True)
    same = torch.equal(a, b)
    note = "  <- past the trained context" if n > model.cfg.max_seq_len else ""
    print(f"  {n:>3} tokens: {'identical' if same else 'DIVERGED'}{note}")
    assert same, (
        f"cached and uncached generation diverged at {n} tokens. "
        "Timing a wrong implementation is meaningless -- fix this first."
    )
print("  GENERATION PATHS EQUIVALENT: PASS")

# -------------------------------------------------------------------- timing
print("\n| generated tokens | uncached | cached | speedup |")
print("|---|---|---|---|")
for n in (64, 128, 256, 512):
    times = {}
    for use_cache in (False, True):
        generate(8, use_cache)                       # warm up
        t0 = time.perf_counter()
        generate(n, use_cache)
        times[use_cache] = time.perf_counter() - t0
    print(f"| {n} | {times[False]:.3f}s | {times[True]:.3f}s | "
          f"**{times[False] / times[True]:.2f}x** |")

print("\nThe gap widens with length: the uncached path re-runs the whole prefix every")
print("step, which is quadratic in generated tokens, while the cached path extends it.")
