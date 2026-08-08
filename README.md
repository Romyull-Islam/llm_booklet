# Build, Train, Deploy: An LLM Booklet

Supporting code for a five-week course that goes from an empty file to a transformer you wrote, trained, and served, then fine-tuned, given tools, and connected to a document collection.

**Nothing here exceeds 256M parameters.** Weeks 1 and 2 run on CPU. Every training run finishes while you are still sitting there.

Read the booklet: **[link to the booklet on your site]**

---

## Reproduce the claims first

The booklet quotes specific numbers. Check them before you trust any of it:

```bash
git clone https://github.com/Romyull-Islam/llm_booklet.git
cd llm_booklet

# Debian, Ubuntu, WSL and Homebrew Python refuse installs into the system
# interpreter (PEP 668). Use a throwaway environment rather than fighting it.
python3 -m venv .venv-verify
source .venv-verify/bin/activate      # Windows: .venv-verify\Scripts\Activate.ps1
pip install torch numpy openai      # openai is needed by the agent-loop check
python verify.py

deactivate && rm -rf .venv-verify     # it existed only to check the numbers
```

Expected output ends with `16/16 checks passed`. It takes about 30 seconds, needs no GPU and no model downloads, and reproduces the KV cache correctness figure, the GQA cache ratio, the MoE parameter accounting, the retrieval metrics, and both agent loops.

If it fails on your machine, that is worth an issue.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # CPU, enough for weeks 1 and 2
cd week1_build
curl -sSL -o input.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

python test_cache.py          # prove the KV cache is correct

# The flags matter: train.py's defaults build a 4.2M model, not the 1.59M one
# the booklet measures. Pass the section 1.5 config to reproduce its numbers.
python train.py --steps 900 --batch-size 16 --seq-len 128 \
                --d-model 192 --n-layers 4 --n-heads 6 --n-kv-heads 2 \
                --out ckpt_dense.pt      # ~6 minutes on one CPU core

python train_words.py --steps 1200 --d-model 192 --n-layers 4 --min-count 3
```

For weeks 3 to 5: `pip install -r requirements-gpu.txt`, then see [COURSE.md](COURSE.md).

## What is in here

| Directory | Contents | Needs |
|---|---|---|
| `week1_build/` | A transformer from primitives: RMSNorm, RoPE, grouped-query attention, KV cache, SwiGLU. Character and word-level training. | CPU |
| `week2_arch/` | Mixture of experts, a toy vision-language model, serving a custom architecture | CPU |
| `week3_finetune/` | LoRA, QLoRA, evaluation, full-fine-tune comparison, merging, GGUF export | GPU |
| `week4_agent/` | Tool calling for both large and small models, the agent loop, MCP, FastAPI, Docker | GPU |
| `week5_rag/` | Chunking, hybrid BM25 and dense retrieval, reranking, retrieval evaluation | GPU |

`COURSE.md` has setup instructions and the run order. `NOTES.md` is a template for recording your own measurements, which is the part that turns this into a portfolio piece.

## Reference results

Two independent runs, so you can see which numbers are properties of the code and
which are properties of the machine. Reproduced by `verify.py` where possible.

| Metric | One CPU core | RTX 4500 Ada (24 GB) |
|---|---|---|
| KV cache, one-shot vs incremental logits | ~1e-7 | ~1e-7 |
| GQA cache reduction at 4096 tokens, `n_heads=8, n_kv_heads=2` | 4.0x (8.4 MB to 2.1 MB) | same |
| GQA cache reduction, `n_heads=6, n_kv_heads=2` (the section 1.5 config) | 3.0x (12.6 MB to 4.2 MB) | same |
| MoE, 8 experts top-2, vs dense at the same architecture | 6.3x parameters, 1.76x active compute | val loss 1.5277 vs 1.5497 dense, at 2.6x wall clock |
| Character model, 1,587,072 params, 900 steps | 360s, val 1.6078 | 5s, val 1.6636 |
| Character model real-word rate **at 128 generated tokens** | 92.1% | 85.0% |
| Character model real-word rate **at 800 generated tokens** | 61.5% | 53.5% |
| Word model, 2,637,120 params, 1200 steps | 575s, val perplexity 75.1 | 136s, val perplexity 78.3 |
| Word model real-word rate, 5 seeds x 5 lengths | **100.0%** | **100.0%** |

**Parameter counts are exact and identical everywhere. Losses, timings and real-word
rates are not** — they move with hardware, seed and, for real-word rate, with how many
tokens you generate. Quote your own, and always say at what length.

**On the KV cache figure.** Do not treat the exact digits as a target. The value is fp32
rounding: it moves with the seed (2.38e-07 to 4.77e-07 across six seeds on one machine),
with the device, and with the data, and in float64 the same test returns ~1e-16. What the
test guarantees is the assertion, `diff < 1e-4`. A real cache bug lands at 1e-2 or larger,
four orders above the threshold, so there is nothing to resolve by comparing digits.

Note also what it does not cover: it proves the cached path agrees with the uncached path,
not that attention is correct. A mistake present in both paths cancels out and passes.

Your numbers will differ. Record them in `NOTES.md`.

## Models used

All Apache 2.0, all under 256M parameters, about 1.2 GB total.

| Role | Model | Params |
|---|---|---|
| Instruct LLM | [SmolLM2-135M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct) | 135M |
| Vision-language | [SmolVLM-256M-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct) | 256M |
| Embeddings | [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) | 22.7M |
| Reranker | [ms-marco-MiniLM-L6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2) | 22.7M |

## Data

No datasets are redistributed here. `week1_build/train.py` downloads Tiny Shakespeare from [karpathy/char-rnn](https://github.com/karpathy/char-rnn) (MIT licensed repository; the underlying text is public domain). Weeks 3 and 5 pull small slices of public Hugging Face datasets at runtime.

## Versions

Library APIs in this stack change often. The code was written against, and argument names verified against:

```
trl 1.9.2 · peft 0.20.0 · sentence-transformers 5.7.0 · langchain 1.3.14 · transformers >= 4.57
```

Last verified: **August 2026**. If something breaks, check the official docs before the booklet, and see Appendix B.1 for the TRL argument renames that break most older tutorials.

## Known limitations

Stated up front so nobody wastes an evening:

- **The 135M model cannot use the OpenAI tool-calling API.** Function calling was trained into SmolLM2's 1.7B variant, not the 135M one. `week4_agent/small_tools.py` is the workaround, and section 4.2 explains it.
- **Your Week 1 model learns form, not meaning.** Every word the word-level model emits is real; the sentences are often still nonsense. 292k training tokens is not enough for semantics. Section 1.7 explains the ceiling and the path past it.
- **`calculator` uses `eval` with restricted builtins.** That is a demonstration, not a sandbox. Do not expose it to untrusted input.
- **Weeks 3 to 5 need a GPU** and were not executed by the author on the machine the booklet was written on. Argument names were verified against the shipping packages; the runs were not. See Appendix E of the booklet for the full disclosure of what was and was not executed.

## Contributing

Issues and pull requests welcome, particularly:

- A number in the booklet that does not reproduce on your hardware
- An API that has drifted since August 2026
- Results from running the GPU chapters, which would let the reference table cover weeks 3 to 5

## License

Code: [MIT](LICENSE). The booklet text is [CC BY 4.0](LICENSE-DOCS).
