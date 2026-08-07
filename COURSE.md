# Build, Train, Deploy: starter code (version 3)

**Size budget: nothing over 256M parameters, no dataset over a few thousand examples.**
Everything loaded at once is under 1 GB of VRAM. Full fine-tuning a 135M model peaks near 2.2 GB.

Every named code file from `booklet.md`, in the folder structure the scripts expect.
Read the booklet alongside these. The code is not a substitute for the explanations.

## Part 1 needs no GPU and no model downloads

`week1_build/` and `week2_arch/` depend only on `torch` and a one-megabyte text file.
They were written and tested on a single CPU core.

```bash
pip install torch
cd week1_build
curl -sSL -o input.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

python test_cache.py     # proves the KV cache is correct: expect ~4.77e-07

# character-level: teaches the mechanics, output is ~90% real words
python train.py --steps 900 --batch-size 16 --seq-len 128 \
                --d-model 192 --n-layers 4 --n-heads 6 --n-kv-heads 2 \
                --eval-every 300 --out ckpt_dense.pt

# word-level: same model, better representation, 100% real words
python train_words.py --steps 1200 --d-model 192 --n-layers 4 --min-count 3
```

Reference run on one CPU core: 1,587,072 parameters, loss 4.1099 to 1.4269 train
and 1.6078 val, 360 seconds. On a 24 GB GPU this is under a minute.

Then grade the output objectively instead of eyeballing it:

```bash
python sample_quality.py ckpt_dense.pt
```

Expect a real-word rate around 90% at short generation lengths for the character
model, falling to roughly 60% by 800 tokens because it was trained at a context of
128. That cliff is the length-extrapolation problem, and seeing it in your own
model takes one minute.

The word-level model scores **100.0% at every generation length**, because it
cannot emit a non-word. Reference run: 2.64M parameters, 575 seconds on one CPU
core, val perplexity 75.1. Section 1.7 of the booklet has both samples.

Mixture of experts is the same script with one flag:

```bash
python train.py --steps 900 --n-experts 8 --d-model 192 --n-layers 4 --out ckpt_moe.pt
```

The toy vision-language model:

```bash
cd ../week2_arch && python tiny_vlm.py
```

## Parts 2 to 4 need the 24 GB card

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv .venv --python 3.12 && source .venv/bin/activate

uv pip install torch --torch-backend=auto
uv pip install "transformers>=4.57" accelerate datasets huggingface_hub \
               trl peft bitsandbytes sentence-transformers \
               "openai>=1.0" fastapi "uvicorn[standard]" gradio \
               numpy pandas matplotlib jupyter python-dotenv

# vLLM goes in its own venv: it pins torch aggressively.
uv venv .venv-vllm --python 3.12 && source .venv-vllm/bin/activate && uv pip install vllm
```

Then `python check_env.py`, and start the downloads:

```bash
hf download Qwen/Qwen3-4B-Instruct-2507
hf download Qwen/Qwen3-VL-4B-Instruct
```

## Run order

| Week | Order | Notes |
|---|---|---|
| 1 | `model.py`, `test_cache.py`, `train.py`, `sample_quality.py`, `word_tok.py`, `train_words.py` | CPU only. Run `test_cache.py` before you trust anything, and `sample_quality.py` before you believe a sample. |
| 2 | `tiny_vlm.py`, `serve_custom.py`, then `02_client.py`, `03_throughput.py`, `04_vlm.py`, `05_ui.py` | The numbered scripts need `vllm serve` running in the other venv |
| 3 | `00_template.py`, `prepare_data.py`, `train_lora.py`, `evaluate.py`, `full_vs_lora.py`, `merge.py` | `train_vlm_lora.py` is the optional bonus |
| 4 | `small_tools.py`, `tools.py`, `agent.py`, `serve.py` | Start with `small_tools.py`: the 135M model cannot use the OpenAI tools API. `rag_agent.py` waits until Week 5 |
| 5 | `chunking.py`, `index.py`, `hybrid.py`, `rerank.py`, `retriever.py`, `generate.py`, `eval_retrieval.py`, then `../week4_agent/rag_agent.py` | the first four are imported by the rest |

## What was executed, not just written

Run on CPU and reproduced in the booklet:

- `week1_build/test_cache.py`: one-shot versus cached-incremental logits agree to **4.77e-07**
- `week1_build/train.py`: full run to 1.4269 train / 1.6078 val loss, dense and MoE paths both
- `week1_build/train_words.py` and `word_tok.py`: full run to 4.3191 val (perplexity 75.1),
  **100% real-word rate over 5 seeds and at 100 through 800 generated tokens**
- `week2_arch/tiny_vlm.py`: correct token splicing and a **non-zero vision encoder gradient** (0.474599)
- `week5_rag/chunking.py`, `hybrid.py`, `eval_retrieval.py`: boundaries, BM25, RRF, recall@k, MRR, nDCG@k
- `week3_finetune/prepare_data.py`: format conversion, audit counts, JSONL round-trip
- `week4_agent/tools.py` and `agent.py`: schema generation plus the full loop including tool errors,
  unknown tools, step budget, and output truncation
- `week4_agent/small_tools.py`: the small-model parser and loop, including comma-containing
  argument values, missing arguments, unknown tools, malformed-line recovery, and the step budget

Everything else needs a GPU and model downloads. Those scripts follow the official
documentation and use argument names checked against the shipping packages
(trl 1.9.2, peft 0.20.0, sentence-transformers 5.7.0, langchain 1.3.14), but your
VRAM figures, throughput, and loss curves will differ. Record your own in `NOTES.md`.

## Security note

The `calculator` tool uses `eval` with restricted builtins. That is a demonstration,
not a sandbox. Before exposing any of this to untrusted input, swap in a proper
expression parser or run tool code inside a container.
