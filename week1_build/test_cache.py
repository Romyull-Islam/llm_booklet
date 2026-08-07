# week1_build/test_cache.py
import torch
from model import Config, Transformer

torch.manual_seed(0)
cfg = Config(vocab_size=65, d_model=128, n_layers=4, n_heads=8, n_kv_heads=2,
             d_ff=352, max_seq_len=128)
m = Transformer(cfg).eval()

seq = torch.randint(0, 65, (1, 24))

# One shot over the whole sequence
full, _, _ = m(seq)

# Prefill 10 tokens, then one at a time
caches = [(None, None)] * cfg.n_layers
step, _, caches = m(seq[:, :10], caches=caches, pos_offset=0)
outs = [step[:, -1]]
for t in range(10, 24):
    step, _, caches = m(seq[:, t:t+1], caches=caches, pos_offset=t)
    outs.append(step[:, -1])
incremental = torch.stack(outs, dim=1)

diff = (full[:, 9:, :] - incremental).abs().max().item()
print(f"max |logit difference|: {diff:.2e}")
assert diff < 1e-4, "KV cache is wrong"
print("KV CACHE NUMERICAL EQUIVALENCE: PASS")
