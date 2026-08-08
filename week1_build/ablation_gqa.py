"""
ablation_gqa.py: exercise 1.5, the grouped-query-attention ablation.

The booklet's section 1.8 checklist and appendix C both require this table, but
the booklet never states the exercise itself. The columns are taken from the
NOTES.md template: n_kv_heads in {6, 2, 1} -> val loss, KV bytes at 4096 tokens,
seconds per step.

Config is the section 1.5 run: 900 steps, batch 16, seq_len 128, d_model 192,
n_layers 4, n_heads 6. Only n_kv_heads changes.
"""
import math, time, torch
from model import Config, Transformer
from train import CharTokenizer, get_batch, estimate_loss, lr_at

STEPS, BATCH, SEQ_LEN = 900, 16, 128
D_MODEL, N_LAYERS, N_HEADS, LR = 192, 4, 6, 3e-3
CACHE_SEQ_LEN, CACHE_BATCH, BYTES_PER_ELEM = 4096, 1, 4     # fp32 KV cache

device = "cuda" if torch.cuda.is_available() else "cpu"
text = open("input.txt", encoding="utf-8").read()
tok = CharTokenizer(text)
ids = torch.tensor(tok.encode(text), dtype=torch.long)
n = int(0.9 * len(ids))
splits = {"train": ids[:n], "val": ids[n:]}


def kv_bytes(n_kv_heads):
    """kv_bytes = 2 (K and V) x n_layers x n_kv_heads x head_dim x seq_len x batch x bytes"""
    head_dim = D_MODEL // N_HEADS
    return (2 * N_LAYERS * n_kv_heads * head_dim
            * CACHE_SEQ_LEN * CACHE_BATCH * BYTES_PER_ELEM)


def run(n_kv_heads):
    torch.manual_seed(0)
    cfg = Config(vocab_size=tok.vocab_size, d_model=D_MODEL, n_layers=N_LAYERS,
                 n_heads=N_HEADS, n_kv_heads=n_kv_heads,
                 d_ff=int(D_MODEL * 8 / 3) // 8 * 8, max_seq_len=SEQ_LEN)
    model = Transformer(cfg).to(device)
    decay = [q for _, q in model.named_parameters() if q.dim() >= 2]
    nod = [q for _, q in model.named_parameters() if q.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1},
                             {"params": nod, "weight_decay": 0.0}],
                            lr=LR, betas=(0.9, 0.95))
    warm = max(1, STEPS // 20)

    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for step in range(STEPS):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, STEPS, LR, warm)
        x, y = get_batch(splits["train"], BATCH, SEQ_LEN, device)
        _, loss, _ = model(x, targets=y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - t0

    ev = estimate_loss(model, splits, BATCH, SEQ_LEN, device)
    params = sum(p.numel() for p in model.parameters())
    return ev["val"], elapsed / STEPS, params


print(f"device {device} | {STEPS} steps, batch {BATCH}, seq_len {SEQ_LEN}, "
      f"d_model {D_MODEL}, n_layers {N_LAYERS}, n_heads {N_HEADS}\n")
LABEL = {6: "6 (multi-head)", 2: "2 (grouped)", 1: "1 (multi-query)"}
print(f"{'n_kv_heads':>16} {'params':>10} {'val loss':>9} "
      f"{'KV @4096':>12} {'reduction':>10} {'s/step':>9}")
base = kv_bytes(N_HEADS)
for h in (6, 2, 1):
    val, sec, params = run(h)
    kb = kv_bytes(h)
    print(f"{LABEL[h]:>16} {params:>10,} {val:>9.4f} "
          f"{kb/1e6:>9.1f} MB {base/kb:>9.1f}x {sec:>9.4f}")
