# week2_finetune/prepare_data.py
import json, random, re
from pathlib import Path

def to_chat(example, system=None):
    """Turn a {'question','answer'} row into TRL conversational prompt-completion format."""
    prompt = []
    if system:
        prompt.append({"role": "system", "content": system})
    prompt.append({"role": "user", "content": example["question"].strip()})
    return {"prompt": prompt,
            "completion": [{"role": "assistant", "content": example["answer"].strip()}]}

def audit(rows):
    """Cheap dataset quality gate. Run this before every training job."""
    report = {"n": len(rows), "empty": 0, "dupes": 0, "too_short": 0, "leaked_prompt": 0}
    seen = set()
    for r in rows:
        q = r["prompt"][-1]["content"]
        a = r["completion"][0]["content"]
        if not q or not a:
            report["empty"] += 1
        key = (q.lower().strip(), a.lower().strip())
        if key in seen:
            report["dupes"] += 1
        seen.add(key)
        if len(a.split()) < 3:
            report["too_short"] += 1
        if q.lower().strip() and q.lower().strip() in a.lower():
            report["leaked_prompt"] += 1
    return report

def split(rows, val_frac=0.1, seed=0):
    rows = list(rows)
    random.Random(seed).shuffle(rows)
    n_val = max(1, int(len(rows) * val_frac))
    return rows[n_val:], rows[:n_val]

def write_jsonl(rows, path):
    Path(path).write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
                          encoding="utf-8")
    return path

if __name__ == "__main__":
    from datasets import load_dataset
    ds = load_dataset("FreedomIntelligence/medical-o1-reasoning-SFT", "en", split="train")
    ds = ds.select(range(2000))          # keep the first run short

    rows = [to_chat({"question": r["Question"], "answer": r["Response"]},
                    system="You are a careful clinical reasoning assistant.")
            for r in ds]

    print("audit:", audit(rows))
    train, val = split(rows, val_frac=0.05)
    write_jsonl(train, "data/train.jsonl")
    write_jsonl(val, "data/val.jsonl")
    print(f"wrote {len(train)} train / {len(val)} val")
