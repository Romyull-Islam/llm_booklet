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
    """Two cheap, honest metrics. Replace with task metrics when you have them."""
    lengths, keyword_hits = [], []
    for r in rows:
        pred = generate(model, tok, r["prompt"])
        gold = r["completion"][0]["content"]
        lengths.append(len(pred.split()))
        gold_terms = {w for w in normalise(gold).split() if len(w) > 5}
        pred_terms = set(normalise(pred).split())
        keyword_hits.append(len(gold_terms & pred_terms) / max(len(gold_terms), 1))
    return {
        "mean_len": sum(lengths) / len(lengths),
        "keyword_recall": sum(keyword_hits) / len(keyword_hits),
    }

rows = [json.loads(l) for l in open("data/val.jsonl")][:50]
tok = AutoTokenizer.from_pretrained(BASE)

base = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map="auto").eval()
before = score(base, tok, rows)
print("BASE      ", before)

tuned = PeftModel.from_pretrained(base, ADAPTER).eval()
after = score(tuned, tok, rows)
print("FINE-TUNED", after)

print("\ndelta keyword_recall:", round(after["keyword_recall"] - before["keyword_recall"], 4))
print("delta mean_len:      ", round(after["mean_len"] - before["mean_len"], 1))
