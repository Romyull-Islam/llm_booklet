# Build, Train, Deploy: starter code

**Size budget: nothing over 256M parameters, no dataset over a few thousand examples.**
Everything loaded at once is under 1 GB of VRAM. Full fine-tuning a 135M model peaks
near 1.8 GB, measured.

Every named code file from the booklet, in the folder structure the scripts expect.
Read the booklet alongside these. The code is not a substitute for the explanations.

## Where to run things

Two rules, and they cause more trouble than any bug in the code:

1. **The repository root is your working directory for weeks 3, 4 and 5.** Those scripts
   read and write `data/` and `models/` as *relative* paths. Launch them by path,
   `python week3_finetune/train_lora.py`, not `cd week3_finetune && python train_lora.py`,
   which fails with `FileNotFoundError: 'data/train.jsonl'`.
2. **Week 1 is the exception.** Its scripts import each other and read `input.txt` from the
   current folder, so you `cd week1_build` and stay there.

## Week 1 needs no GPU and no model downloads

`week1_build/` depends only on `torch` and a one-megabyte text file. It was written and
tested on a single CPU core.

```bash
# Debian, Ubuntu, WSL and Homebrew Python refuse installs into the system interpreter
# (PEP 668). Use a virtual environment rather than fighting it.
python3 -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install torch numpy openai      # openai is needed by verify.py's agent check

cd week1_build
curl -sSL -o input.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

python test_cache.py     # proves the KV cache is correct

# character-level: teaches the mechanics, output is ~85-92% real words at 128 tokens
python train.py --steps 900 --batch-size 16 --seq-len 128 \
                --d-model 192 --n-layers 4 --n-heads 6 --n-kv-heads 2 \
                --eval-every 300 --out ckpt_dense.pt

# word-level: same model, better representation, 100% real words
python train_words.py --steps 1200 --d-model 192 --n-layers 4 --min-count 3
```

`test_cache.py` prints a difference of order **1e-7**. The exact digits are fp32 rounding
and move with the seed, the device and the dtype, so do not compare them against anyone
else's. What matters is the assertion, `diff < 1e-4`. A genuinely broken cache lands at
1e-2 or worse, four orders above the threshold.

Reference run on one CPU core: 1,587,072 parameters, loss 4.1099 to 1.4269 train and
1.6078 val, 360 seconds. On a 24 GB GPU the same run is about 5 seconds.

Then grade the output objectively instead of eyeballing it:

```bash
python sample_quality.py ckpt_dense.pt        # 400 tokens by default
python quality_vs_length.py ckpt_dense.pt     # the full curve, 3 seeds per row
```

**Always state the generation length with a real-word rate.** The character model scores
around 85-92% at 128 tokens and falls to roughly 53-61% by 800, because it was trained at
a context of 128. That cliff is the length-extrapolation problem, and seeing it in your own
model takes one minute.

The word-level model scores **100.0% at every generation length**, because it cannot emit a
non-word. Reference run: 2,637,120 parameters, 575 seconds on one CPU core, val perplexity
75.1. Section 1.7 of the booklet has both samples.

The exercises from section 1.5:

```bash
python ablation_gqa.py --steps 500            # the grouped-query attention table
python gen_timing.py ckpt_dense.pt            # cached vs uncached generation
```

### Scaling up: same code, better-matched data

Tiny Shakespeare is too small for the model to learn what words *mean*. TinyStories is the
canonical demonstration that this is a data problem, not an architecture problem:

```bash
python get_tinystories.py --stories 200000    # builds tinystories.txt from the Hub

# --n-heads 8 is required: at d_model 512 the default 6 heads is not a divisor
# (512 % 6 != 0) and the model will not construct.
python train_words.py --data tinystories.txt --steps 20000 \
                      --d-model 512 --n-layers 8 --n-heads 8 --min-count 5 \
                      --seq-len 256 --batch-size 32 --prompt "Once upon a time"
```

About 26.7M parameters, still far under the budget. Verified: 2,000 steps on one GPU took
4m45s and reached val perplexity 7.9, producing coherent, consistent short stories.

## Week 2

Mixture of experts is the same script with one flag. **Pass `--n-heads 6`**: the default is
8, and without it you compare two different architectures and call it an MoE result.

```bash
python train.py --steps 900 --n-experts 8 --d-model 192 --n-layers 4 \
                --n-heads 6 --out ckpt_moe.pt

# the matched dense baseline, or you have no comparison
python train.py --steps 900 --d-model 192 --n-layers 4 --n-heads 6 \
                --out ckpt_dense_base.pt

# routing collapse: run it again with the load-balancing term switched off
python train.py --steps 900 --n-experts 8 --d-model 192 --n-layers 4 \
                --n-heads 6 --aux-loss-coef 0.0 --out /tmp/ckpt_moe_noaux.pt
```

The toy vision-language model, CPU only:

```bash
cd ../week2_arch && python tiny_vlm.py
```

**Sections 2.4 to 2.6 need a GPU and about 780 MB of downloads** (`SmolLM2-135M-Instruct`
and `SmolVLM-256M-Instruct`), so they are not part of the CPU-only half.

## Weeks 3 to 5 want a GPU

8 GB is plenty. Measured peaks on this code: LoRA training 1.1-1.2 GB, full fine-tuning
1.8 GB, VLM inference 0.8 GB.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv .venv --python 3.12 && source .venv/bin/activate

uv pip install torch --torch-backend=auto
uv pip install "transformers>=4.57" accelerate datasets huggingface_hub \
               trl peft bitsandbytes sentence-transformers torchvision \
               "openai>=1.0" fastapi "uvicorn[standard]" "gradio<7" \
               onnx onnxscript onnxruntime \
               numpy pandas matplotlib jupyter python-dotenv

# vLLM goes in its own venv: it pins torch aggressively.
uv venv .venv-vllm --python 3.12 && source .venv-vllm/bin/activate && uv pip install vllm
deactivate && source .venv/bin/activate
```

`torchvision` is needed by the transformers v5 image processor, and the ONNX packages by
section 2.4's export. Neither is optional if you intend to run those sections.

**Virtual environments are not relocatable.** The scripts in `.venv/bin/` hard-code the
absolute path they were built at, so if you move or rename this folder afterwards every
command breaks with `bad interpreter: No such file or directory`. Delete and recreate both
environments if you move it.

Then `python check_env.py`, and start the downloads:

```bash
# About 1.15 GB of weights. --exclude matters: these repos also ship ONNX and GGUF
# conversions that nothing here uses, and without it you pull about 6.7 GB.
for m in HuggingFaceTB/SmolLM2-135M-Instruct \
         HuggingFaceTB/SmolVLM-256M-Instruct \
         sentence-transformers/all-MiniLM-L6-v2 \
         cross-encoder/ms-marco-MiniLM-L6-v2; do
  hf download "$m" --exclude "*.onnx" "*.onnx_data" "*.gguf" "*.msgpack" "*.h5"
done
```

| Role | Model | Params | Weights |
|---|---|---|---|
| Instruct LLM | `HuggingFaceTB/SmolLM2-135M-Instruct` | 135M | 0.27 GB |
| Vision-language | `HuggingFaceTB/SmolVLM-256M-Instruct` | 256M | 0.51 GB |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` | 22.7M | 0.18 GB |
| Reranker | `cross-encoder/ms-marco-MiniLM-L6-v2` | 22.7M | 0.18 GB |

## Run order

| Week | Order | Notes |
|---|---|---|
| 1 | `model.py`, `test_cache.py`, `train.py`, `sample_quality.py`, `quality_vs_length.py`, `word_tok.py`, `train_words.py`, `ablation_gqa.py`, `gen_timing.py` | CPU only. Run `test_cache.py` before you trust anything, and `sample_quality.py` before you believe a sample. |
| 2 | `tiny_vlm.py`, `serve_custom.py`, `export_onnx.py`, then `02_client.py`, `03_throughput.py`, `04_vlm.py`, `05_ui.py` | The numbered scripts need `vllm serve` running in the other venv. `serve_custom.py` runs from `week2_arch/`; the rest of weeks 3-5 run from the repo root. |
| 3 | `00_template.py`, `prepare_data.py`, `check_labels.py`, `train_lora.py`, `evaluate.py`, `full_vs_lora.py`, `merge.py` | `train_vlm_lora.py` is the optional bonus. Run all of these from the repo root. |
| 4 | `small_tools.py`, `measure_tools.py`, `tools.py`, `agent.py`, `serve.py` | Start with `small_tools.py`: the 135M model cannot use the OpenAI tools API. `rag_agent.py` waits until Week 5 |
| 5 | `build_corpus.py`, then `chunking.py`, `index.py`, `hybrid.py`, `rerank.py`, `retriever.py`, `generate.py`, `eval_retrieval.py`, then `../week4_agent/rag_agent.py` | `build_corpus.py` first: `eval_retrieval.py` needs `data/chunks.json` and `data/eval_questions.json`. The middle four are imported by the rest. |

**The agent scripts need two extra vLLM flags.** `agent.py`, `agent_langchain.py`,
`serve.py` and `rag_agent.py` all send a `tools` array, and vLLM rejects that with HTTP 400
unless you start it with `--enable-auto-tool-choice --tool-call-parser hermes`. Even then,
SmolLM2-135M has no tool support in its chat template and will not emit a tool call — which
is exactly why section 4.2 exists and why `small_tools.py` is the path that works at this
size. Use `rag_agent_small.py` for a capstone that actually fires its tools.

## What was executed, not just written

Run on CPU and reproduced in the booklet:

- `week1_build/test_cache.py`: one-shot versus cached-incremental logits agree to order **1e-7**
- `week1_build/train.py`: full run to 1.4269 train / 1.6078 val loss, dense and MoE paths both
- `week1_build/train_words.py` and `word_tok.py`: full run to 4.3191 val (perplexity 75.1),
  **100% real-word rate over 5 seeds and at 100 through 800 generated tokens**
- `week2_arch/tiny_vlm.py`: correct token splicing and a **non-zero vision encoder gradient**
- `week5_rag/chunking.py`, `hybrid.py`, `eval_retrieval.py`: boundaries, BM25, RRF, recall@k, MRR, nDCG@k
- `week3_finetune/prepare_data.py`: format conversion, audit counts, JSONL round-trip
- `week4_agent/tools.py` and `agent.py`: schema generation plus the full loop including tool errors,
  unknown tools, step budget, and output truncation
- `week4_agent/small_tools.py`: the small-model parser and loop, including comma-containing
  argument values, missing arguments, unknown tools, malformed-line recovery, and the step budget

Run `python verify.py` to reproduce all of the above in about 30 seconds, with no GPU and no
downloads. It ends with `16/16 checks passed`.

Everything else needs a GPU and model downloads. Argument names are checked against the
shipping packages (trl 1.9.2, peft 0.20.0, sentence-transformers 5.7.0, langchain 1.3.14),
but your VRAM figures, throughput and loss curves will differ. Record your own in `NOTES.md`.

## Security note

The `calculator` tool uses `eval` with restricted builtins. That is a demonstration,
not a sandbox. Before exposing any of this to untrusted input, swap in a proper
expression parser or run tool code inside a container.
