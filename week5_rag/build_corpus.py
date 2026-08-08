"""
build_corpus.py: produce data/chunks.json and data/eval_questions.json.

Section 5.6 needs both files and the booklet ships neither, so eval_retrieval.py
cannot run out of the box. The corpus here is the booklet itself, split into one
document per `##` section, then chunked with the section 5.2 chunker.

LABELLING METHOD, stated so the numbers can be judged. Section 5.6 says to write
the questions by hand and label the chunk IDs that genuinely answer them. The
questions below are hand-written. The gold IDs are derived by locating an
`anchor` string -- an exact phrase from the passage that answers the question --
and marking every chunk containing it. That is string-grounded rather than
eyeballed, it is reproducible, and it fails loudly (a question with no matching
chunk is reported rather than silently scoring zero).
"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chunking import build_corpus                                   # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
SRC = os.path.join(ROOT, "data", "booklet.md")
OUT_CHUNKS = os.path.join(ROOT, "data", "chunks.json")
OUT_EVAL = os.path.join(ROOT, "data", "eval_questions.json")

text = open(SRC, encoding="utf-8").read()

# One document per `##` section, so doc_id and title mean something for citations.
parts = re.split(r"\n(?=##\s)", text)
docs = []
for i, part in enumerate(parts):
    m = re.match(r"##+\s*(.+)", part)
    title = m.group(1).strip() if m else "Front matter"
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:48] or f"sec-{i}"
    docs.append({"id": f"{i:02d}-{slug}", "title": title,
                 "url": f"#{slug}", "text": part})

records = build_corpus(docs, chunk_chars=1200, overlap=200)
json.dump(records, open(OUT_CHUNKS, "w"), indent=1)
print(f"{len(docs)} documents -> {len(records)} chunks")
print(f"chunk chars: min {min(len(r['text']) for r in records)}, "
      f"max {max(len(r['text']) for r in records)}, "
      f"mean {sum(len(r['text']) for r in records)//len(records)}")

# Hand-written questions. `anchor` is an exact phrase from the answering passage.
QA = [
    ("What is the maximum parameter count for any model in this booklet?",
     "No model here exceeds 256M parameters"),
    ("What replaced LayerNorm in modern language models?",
     "LayerNorm without the mean subtraction"),
    ("Why does RoPE rotate instead of add position vectors?",
     "depends only on (m - n)"),
    ("What is the formula for KV cache size in bytes?",
     "kv_bytes = 2 (K and V) x n_layers x n_kv_heads"),
    ("Why is is_causal set to T > 1 rather than always True?",
     "Applying a causal mask to a length-1 query"),
    ("What tolerance proves the KV cache is correct?",
     "max |logit difference|: 4.77e-07"),
    ("Why does SwiGLU use d_ff of about 2.75 times d_model?",
     "three matrices instead of two"),
    ("What is the difference between pre-norm and post-norm?",
     "clean unnormalised path from the loss"),
    ("What betas should AdamW use for language modelling?",
     "betas=(0.9, 0.95)"),
    ("Why should weight decay be skipped on 1-D parameters?",
     "Decaying norm scales and biases toward zero"),
    ("How do you stop a character model from inventing words?",
     "Give it word tokens and misspelling becomes impossible"),
    ("What are the four ordered questions to ask when output is bad?",
     "Is the representation wrong?"),
    ("What does the mixture-of-experts auxiliary loss prevent?",
     "the router collapses onto a couple of favourite experts"),
    ("What auxiliary loss value means the router is perfectly balanced?",
     "perfectly balanced value is exactly"),
    ("What fraction of a transformer's parameters are MLP rather than attention?",
     "two thirds of a modern transformer's parameters are MLP"),
    ("What aspect ratio should d_model to n_layers have?",
     "sits between 32 and 128"),
    ("How many tokens per parameter does Chinchilla recommend?",
     "20 tokens per parameter"),
    ("Why can't vLLM serve a custom architecture?",
     "hand-written, hand-optimised implementations"),
    ("What are the three parts of a vision-language model?",
     "vision encoder -> projector"),
    ("Which knob do you turn first when a VLM runs out of memory?",
     "lower this first when you OOM"),
    ("When should you use RAG instead of fine-tuning?",
     "fine-tuning teaches behaviour, retrieval supplies knowledge"),
    ("What is the recommended relationship between lora_alpha and r?",
     "alpha = 2 * r"),
    ("Which modules should LoRA adapters attach to for a Llama architecture?",
     "gate_proj, up_proj, down_proj"),
    ("What does QLoRA add on top of LoRA?",
     "4-bit NF4 and dequantise on the fly"),
    ("Why must the training string match the inference string exactly?",
     "must be byte-identical to the string you infer on"),
    ("What does add_generation_prompt=True actually do?",
     "it hands the model the floor"),
    ("Why can't SmolLM2-135M use the OpenAI tools API?",
     "trained into the 1.7B variant"),
    ("What three things make tool calling work on a small model?",
     "A format the model can copy"),
    ("What does MCP standardise?",
     "standardises how tools are exposed to models"),
    ("How much should RAG chunks overlap?",
     "Overlap** by 10 to 20 percent"),
    ("What chunk size should you target for RAG?",
     "Target 500 to 1500 characters"),
    ("Why does all-MiniLM-L6-v2 not need a query prefix?",
     "symmetric: queries and documents are embedded the same way"),
    ("What does reciprocal rank fusion combine?",
     "rrf"),
    ("How many hand-labelled questions does the booklet recommend for eval?",
     "30 to 50 questions"),
]

eval_set, missing = [], []
for question, anchor in QA:
    gold = [r["chunk_id"] for r in records if anchor in r["text"]]
    if not gold:
        missing.append((question, anchor))
        continue
    eval_set.append({"question": question, "anchor": anchor, "gold_chunk_ids": gold})

json.dump(eval_set, open(OUT_EVAL, "w"), indent=1)
print(f"{len(eval_set)} eval questions written, "
      f"mean {sum(len(e['gold_chunk_ids']) for e in eval_set)/len(eval_set):.2f} gold chunks each")
if missing:
    print(f"\n{len(missing)} anchors did not match any chunk and were dropped:")
    for q, a in missing:
        print(f"  - {q!r} anchor={a!r}")
