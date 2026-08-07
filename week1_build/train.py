"""
train.py: pretrain the model in model.py on raw text, from random weights.

This is real pretraining, just small. The loop is identical in structure to the
one that trains a 400B model: batch, forward, loss, backward, clip, step, schedule.
"""
import math
import time
import argparse
import torch
from model import Config, Transformer


# ------------------------------------------------------------------- tokeniser

class CharTokenizer:
    """The simplest tokeniser that works. Swap for BPE once the loop runs."""

    def __init__(self, text):
        self.chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.itos = {i: c for c, i in self.stoi.items()}
        self.vocab_size = len(self.chars)

    def encode(self, s):
        return [self.stoi[c] for c in s if c in self.stoi]

    def decode(self, ids):
        return "".join(self.itos[int(i)] for i in ids)


def get_batch(data, batch_size, seq_len, device):
    """Sample random windows. targets are inputs shifted left by one."""
    ix = torch.randint(len(data) - seq_len - 1, (batch_size,))
    x = torch.stack([data[i:i + seq_len] for i in ix])
    y = torch.stack([data[i + 1:i + 1 + seq_len] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, splits, batch_size, seq_len, device, iters=20):
    model.eval()
    out = {}
    for name, data in splits.items():
        losses = torch.zeros(iters)
        for k in range(iters):
            x, y = get_batch(data, batch_size, seq_len, device)
            _, loss, _ = model(x, targets=y)
            losses[k] = loss.item()
        out[name] = losses.mean().item()
    model.train()
    return out


def lr_at(step, total, base_lr, warmup):
    """Linear warmup then cosine decay to 10 percent. The standard schedule."""
    if step < warmup:
        return base_lr * (step + 1) / warmup
    progress = (step - warmup) / max(total - warmup, 1)
    min_lr = 0.1 * base_lr
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="input.txt")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--n-layers", type=int, default=6)
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--n-kv-heads", type=int, default=2)
    p.add_argument("--n-experts", type=int, default=1)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--out", default="ckpt.pt")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    text = open(args.data, encoding="utf-8").read()
    tok = CharTokenizer(text)
    ids = torch.tensor(tok.encode(text), dtype=torch.long)
    n = int(0.9 * len(ids))
    splits = {"train": ids[:n], "val": ids[n:]}
    print(f"device {device} | chars {len(text):,} | vocab {tok.vocab_size} | "
          f"train {len(splits['train']):,} val {len(splits['val']):,}")

    cfg = Config(
        vocab_size=tok.vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        n_kv_heads=args.n_kv_heads,
        d_ff=int(args.d_model * 8 / 3) // 8 * 8,
        max_seq_len=args.seq_len,
        n_experts=args.n_experts,
    )
    model = Transformer(cfg).to(device)
    rep = model.param_report()
    print(f"params total {rep['total']:,} | active per token {rep['active_per_token']:,}")

    # No weight decay on norms and embeddings. Standard, and it matters.
    decay = [p for n_, p in model.named_parameters() if p.dim() >= 2]
    nodecay = [p for n_, p in model.named_parameters() if p.dim() < 2]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": 0.1}, {"params": nodecay, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.95), eps=1e-8,
    )

    warmup = max(1, args.steps // 20)
    t0 = time.time()
    for step in range(args.steps):
        lr = lr_at(step, args.steps, args.lr, warmup)
        for g in opt.param_groups:
            g["lr"] = lr

        x, y = get_batch(splits["train"], args.batch_size, args.seq_len, device)
        _, loss, _ = model(x, targets=y)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % args.eval_every == 0 or step == args.steps - 1:
            ev = estimate_loss(model, splits, args.batch_size, args.seq_len, device)
            print(f"step {step:5d} | train {ev['train']:.4f} | val {ev['val']:.4f} | "
                  f"lr {lr:.2e} | grad {gnorm:.2f} | {time.time()-t0:.0f}s")

    torch.save({"model": model.state_dict(), "cfg": cfg, "chars": tok.chars}, args.out)
    print(f"saved {args.out}")

    prompt = torch.tensor([tok.encode("KING RICHARD")], device=device)
    out = model.generate(prompt, 300, temperature=0.8, top_k=40)
    print("\n--- sample ---")
    print(tok.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
