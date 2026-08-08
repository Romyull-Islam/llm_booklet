"""Same model, same loop, word-level tokens instead of characters."""
import math, time, argparse, torch
from model import Config, Transformer
from word_tok import WordTokenizer
from train import get_batch, estimate_loss, lr_at

p = argparse.ArgumentParser()
# --data is what makes the TinyStories scale-up at the end of section 1.7 runnable.
# Without it this script hardcoded input.txt and the booklet's command died with
# "unrecognized arguments: --data tinystories.txt".
p.add_argument("--data", default="input.txt")
p.add_argument("--steps", type=int, default=1200)
p.add_argument("--batch-size", type=int, default=16)
p.add_argument("--seq-len", type=int, default=128)
p.add_argument("--d-model", type=int, default=192)
p.add_argument("--n-layers", type=int, default=4)
# n_heads was hardcoded to 6, so --d-model 512 tripped the
# `assert d_model % n_heads == 0` in Attention (512 % 6 = 2). Section 2.2's own
# rule wants head_dim of 64 or 128: at d_model 512 that means n_heads 8 or 4.
p.add_argument("--n-heads", type=int, default=6)
p.add_argument("--n-kv-heads", type=int, default=2)
p.add_argument("--min-count", type=int, default=3)
p.add_argument("--lr", type=float, default=2e-3)
p.add_argument("--eval-every", type=int, default=200)
p.add_argument("--prompt", default="KING RICHARD")
p.add_argument("--out", default="ckpt_words.pt")
a = p.parse_args()

# The booklet quotes GPU wall clock for the TinyStories run, but this script used
# to pin everything to "cpu" and never move the model, so a GPU was unreachable.
device = "cuda" if torch.cuda.is_available() else "cpu"

text = open(a.data, encoding="utf-8").read()
tok = WordTokenizer(text, min_count=a.min_count)
ids = torch.tensor(tok.encode(text), dtype=torch.long)
n = int(0.9 * len(ids))
splits = {"train": ids[:n], "val": ids[n:]}
print(f"device {device} | corpus {a.data} | vocab {tok.vocab_size} | "
      f"tokens {len(ids):,} | coverage {tok.coverage:.4f}")

cfg = Config(vocab_size=tok.vocab_size, d_model=a.d_model, n_layers=a.n_layers,
             n_heads=a.n_heads, n_kv_heads=a.n_kv_heads,
             d_ff=int(a.d_model*8/3)//8*8, max_seq_len=a.seq_len)
model = Transformer(cfg).to(device)
print("params:", model.param_report())

decay = [q for _, q in model.named_parameters() if q.dim() >= 2]
nod = [q for _, q in model.named_parameters() if q.dim() < 2]
opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1},
                         {"params": nod, "weight_decay": 0.0}],
                        lr=a.lr, betas=(0.9, 0.95))
warm = max(1, a.steps // 20)
t0 = time.time()
for step in range(a.steps):
    lr = lr_at(step, a.steps, a.lr, warm)
    for g in opt.param_groups: g["lr"] = lr
    x, y = get_batch(splits["train"], a.batch_size, a.seq_len, device)
    _, loss, _ = model(x, targets=y)
    opt.zero_grad(set_to_none=True); loss.backward()
    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    if step % a.eval_every == 0 or step == a.steps - 1:
        ev = estimate_loss(model, splits, a.batch_size, a.seq_len, device, iters=10)
        print(f"step {step:5d} | train {ev['train']:.4f} | val {ev['val']:.4f} | "
              f"ppl {math.exp(ev['val']):.1f} | grad {gn:.2f} | {time.time()-t0:.0f}s", flush=True)

torch.save({"model": model.state_dict(), "cfg": cfg, "itos": tok.itos}, a.out)
torch.manual_seed(0)
# ban_tokens=[0] is the <unk> ban the booklet requires: index 0 is <unk> and it
# would otherwise be emitted as literal '<unk>' text, which is not a real word.
out = model.generate(torch.tensor([tok.encode(a.prompt)], device=device), 120,
                     temperature=0.8, top_k=40, ban_tokens=[0])
print("\n--- sample ---")
print(tok.decode(out[0].tolist()))
