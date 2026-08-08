# week3_finetune/evaluate.py
import json, re, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE = "HuggingFaceTB/SmolLM2-135M-Instruct"
ADAPTER = "models/smollm2-135m-lora/final"

def normalise(s):
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()

def generate(model, tok, prompt_messages, max_new_tokens=256):
    text = tok.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                             do_sample=False,               # greedy: evaluation must be deterministic
                             pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)

def score(model, tok, rows):
    """Cheap keyword overlap, reported as precision, recall AND F1.

    Recall alone is monotone in output length: padding an answer with extra words
    can only raise it. A model that learns to be concise then looks like a
    regression. Always report a precision term next to a recall term.
    """
    lengths, prec, rec = [], [], []
    for r in rows:
        pred = generate(model, tok, r["prompt"])
        gold = r["completion"][0]["content"]
        lengths.append(len(pred.split()))
        gold_terms = {w for w in normalise(gold).split() if len(w) > 5}
        pred_terms = {w for w in normalise(pred).split() if len(w) > 5}
        hit = len(gold_terms & pred_terms)
        rec.append(hit / max(len(gold_terms), 1))
        prec.append(hit / max(len(pred_terms), 1))
    mean = lambda xs: sum(xs) / max(len(xs), 1)
    r_, p_ = mean(rec), mean(prec)
    return {
        "mean_len": mean(lengths),
        "keyword_recall": r_,
        "keyword_precision": p_,
        "keyword_f1": 2 * p_ * r_ / (p_ + r_) if (p_ + r_) else 0.0,
    }

rows = [json.loads(l) for l in open("data/val.jsonl")][:40]
tok = AutoTokenizer.from_pretrained(BASE)

base = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map="auto").eval()
before = score(base, tok, rows)
print("BASE      ", before)

tuned = PeftModel.from_pretrained(base, ADAPTER).eval()
after = score(tuned, tok, rows)
print("FINE-TUNED", after)

print()
for k in ("keyword_f1", "keyword_precision", "keyword_recall", "mean_len"):
    print(f"delta {k:<18} {after[k] - before[k]:+.4f}")
print("\nRead F1 first. If recall fell while precision rose, the model got more "
      "concise, which is usually an improvement, not a regression.")
