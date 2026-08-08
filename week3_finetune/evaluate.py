# week2_finetune/evaluate.py
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
    """Two cheap, honest metrics. Replace with task metrics when you have them.

    keyword_recall alone is LENGTH-BIASED: it has no precision term, so a model
    that learns to answer concisely scores worse even when it improved. Precision
    and F1 are reported alongside so the length effect is visible rather than
    silently folded into the headline number.
    """
    lengths, keyword_hits, precisions, f1s = [], [], [], []
    for r in rows:
        pred = generate(model, tok, r["prompt"])
        gold = r["completion"][0]["content"]
        lengths.append(len(pred.split()))
        gold_terms = {w for w in normalise(gold).split() if len(w) > 5}
        pred_terms = set(normalise(pred).split())
        overlap = len(gold_terms & pred_terms)
        recall = overlap / max(len(gold_terms), 1)
        # Precision is over the same class of terms, so the two are comparable.
        pred_long = {w for w in pred_terms if len(w) > 5}
        precision = overlap / max(len(pred_long), 1)
        keyword_hits.append(recall)
        precisions.append(precision)
        f1s.append(0.0 if recall + precision == 0
                   else 2 * recall * precision / (recall + precision))
    n = len(rows)
    return {
        "mean_len": sum(lengths) / n,
        "keyword_recall": sum(keyword_hits) / n,
        "keyword_precision": sum(precisions) / n,
        "keyword_f1": sum(f1s) / n,
    }

rows = [json.loads(l) for l in open("data/val.jsonl")][:50]
tok = AutoTokenizer.from_pretrained(BASE)

base = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map="auto").eval()
before = score(base, tok, rows)
print("BASE      ", before)

tuned = PeftModel.from_pretrained(base, ADAPTER).eval()
after = score(tuned, tok, rows)
print("FINE-TUNED", after)

print()
for k in ("keyword_recall", "keyword_precision", "keyword_f1", "mean_len"):
    print(f"delta {k:<18} {after[k] - before[k]:+.4f}")
