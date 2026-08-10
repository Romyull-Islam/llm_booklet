"""
build_corpus.py: turn documents into data/chunks.json, and an eval set if it can.

Section 5.6 needs `data/chunks.json`, and `eval_retrieval.py` also needs
`data/eval_questions.json`. This produces both.

    python week5_rag/build_corpus.py --src data/booklet.md      # one file
    python week5_rag/build_corpus.py --src ~/notes              # a folder of .md/.txt

**Use your own documents.** A retriever over a corpus you know well is worth ten of
one over someone else's, because you can tell instantly whether an answer is right.

If you have nothing in mind, save the booklet as `data/booklet.md` and use that: it
is a real document, you have just read it, and the questions at the bottom of this
file are anchored to it, so you get an eval set without writing one.

LABELLING METHOD, stated so the numbers can be judged. Section 5.6 says to write the
questions by hand and label the chunk IDs that answer them. The questions below are
hand-written. Gold IDs come from locating an `anchor`, an exact phrase from the
answering passage, and marking every chunk containing it. It is reproducible, and it
fails loudly: an anchor matching nothing is reported rather than scoring zero in
silence. Edit the source document and some anchors will go stale; that report is the
signal.
"""
import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chunking import build_corpus                                   # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

ap = argparse.ArgumentParser()
ap.add_argument("--src", default=os.path.join(ROOT, "data", "booklet.md"),
                help="a .md/.txt file, or a folder containing them")
ap.add_argument("--chunk-chars", type=int, default=1200)
ap.add_argument("--overlap", type=int, default=200)
args = ap.parse_args()

SRC = args.src
OUT_CHUNKS = os.path.join(ROOT, "data", "chunks.json")
OUT_EVAL = os.path.join(ROOT, "data", "eval_questions.json")

if not os.path.exists(SRC):
    sys.exit(
        f"No documents found at {SRC}\n\n"
        "Week 5 needs a corpus. This repository does not ship one, because the\n"
        "point of the chapter is to retrieve over documents you know.\n\n"
        "Either:\n\n"
        "  1. Point --src at a folder of your own .md or .txt files:\n"
        "       python week5_rag/build_corpus.py --src /path/to/your/notes\n"
        "     Any documents you have read will do: a runbook, project docs, notes.\n"
        "     You will write your own eval questions, which section 5.6 explains.\n\n"
        "  2. Or use the booklet itself. Save it as data/booklet.md and rerun with\n"
        "     no arguments:\n"
        "       mkdir -p data && cp /path/to/booklet.md data/booklet.md\n"
        "       python week5_rag/build_corpus.py\n"
        "     The 34 questions in this file are anchored to the booklet's text, so\n"
        "     this path gives you a labelled eval set without writing one."
    )

docs = []
if os.path.isdir(SRC):
    for dirpath, _, names in os.walk(SRC):
        for fn in sorted(names):
            if not fn.lower().endswith((".md", ".txt")):
                continue
            path = os.path.join(dirpath, fn)
            rel = os.path.relpath(path, SRC)
            docs.append({"id": re.sub(r"[^a-z0-9]+", "-", rel.lower()).strip("-"),
                         "title": rel, "url": rel,
                         "text": open(path, encoding="utf-8", errors="replace").read()})
    if not docs:
        sys.exit(f"No .md or .txt files under {SRC}")
else:
    text = open(SRC, encoding="utf-8").read()
    # One document per `##` section, so doc_id and title mean something for citations.
    for i, part in enumerate(re.split(r"\n(?=##\s)", text)):
        m = re.match(r"##+\s*(.+)", part)
        title = m.group(1).strip() if m else "Front matter"
        slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:48] or f"sec-{i}"
        docs.append({"id": f"{i:02d}-{slug}", "title": title,
                     "url": f"#{slug}", "text": part})

os.makedirs(os.path.join(ROOT, "data"), exist_ok=True)
records = build_corpus(docs, chunk_chars=args.chunk_chars, overlap=args.overlap)
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
     "Expect a difference of order **1e-7**"),
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
     "its chat template contains no `tools` block"),
    ("What three things make tool calling work on a small model?",
     "A format the model can copy"),
    ("What does MCP standardise?",
     "standardises how tools are exposed to models"),
    ("How much should RAG chunks overlap?",
     "Overlap** by 10 to 20 percent"),
    ("What chunk size should you target for RAG?",
     "Target 500 to 1500 characters"),
    ("Why does all-MiniLM-L6-v2 not need a query prefix?",
     "all-MiniLM-L6-v2 is symmetric: no query prefix"),
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

if not eval_set:
    print(f"\nNone of the built-in anchors matched {SRC}, which is expected for your own\n"
          f"documents. Write 30 to 50 questions and save them as {OUT_EVAL}:\n"
          '  [{"question": "...", "gold_chunk_ids": ["<chunk_id>", ...]}, ...]\n'
          "Section 5.6 explains why hand-labelling is the highest-leverage hour in Week 5.")
    sys.exit(0)

json.dump(eval_set, open(OUT_EVAL, "w"), indent=1)
print(f"{len(eval_set)} eval questions written, "
      f"mean {sum(len(e['gold_chunk_ids']) for e in eval_set)/len(eval_set):.2f} gold chunks each")
if missing:
    print(f"\n{len(missing)} anchors did not match any chunk and were dropped:")
    for q, a in missing:
        print(f"  - {q!r} anchor={a!r}")
