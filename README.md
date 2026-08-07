# Build, Train, Deploy: An LLM Booklet

Supporting code for a five-week course that goes from an empty file to a transformer you wrote, trained, and served, then fine-tuned, given tools, and connected to a document collection.

**Nothing here exceeds 256M parameters.** Weeks 1 and 2 run on CPU. Every training run finishes while you are still sitting there.

Read the booklet: **[link to the booklet on your site]**

---

## Reproduce the claims first

The booklet quotes specific numbers. Check them before you trust any of it:

```bash
git clone https://github.com/YOUR-USERNAME/llm-booklet
cd llm-booklet
pip install torch numpy
python verify.py
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
python train.py --steps 900   # character-level, about 6 minutes on one CPU core
python train_words.py --steps 1200   # word-level, 100% real words
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

Measured on **one CPU core**, reproduced by `verify.py` where possible:

| Metric | Value |
|---|---|
| KV cache, one-shot vs incremental logits | `4.77e-07` max difference |
| GQA cache reduction at 4096 tokens | 4.0x (8.4 MB to 2.1 MB) |
| MoE, 8 experts top-2 | 6.3x parameters for 1.76x active compute |
| Character model, 1.59M params, 900 steps | 360s, val loss 1.6078, 92.1% real words |
| Word model, 2.64M params, 1200 steps | 575s, val perplexity 75.1, **100% real words** |

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
