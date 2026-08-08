# Course notebook

Record every measurement here. These become your CV bullets and, if you go the
academia route, your methods section.

**Always write down the conditions with the number**: which device, how many CPU
threads, which seed, and for any generation metric, how many tokens you generated.
A figure without its configuration cannot be compared to anything, including your
own next run.

## Week 0: environment

- machine: CPU cores / GPU model / VRAM:
- `python check_env.py` output, pasted:
- torch, transformers, trl, peft, sentence-transformers versions:
- `python verify.py` result (expect `16/16 checks passed`):

## Week 1: build and train

- machine used for Week 1, and thread count:
- parameter count reported by `train.py`:
- `test_cache.py` difference (the assertion is `< 1e-4`; the digits are rounding
  and will not match anyone else's machine):
- first training loss vs `ln(vocab_size)`:
- final train / val loss, wall clock:
- seconds per step:
- sample output at the end, pasted:
- real-word rate from `sample_quality.py`, **and at how many generated tokens**:

**Quality versus generation length** (`quality_vs_length.py`, 3 seeds per row):

| generated tokens | real-word rate |
|---|---|
| 100 |  |
| 128 (trained context) |  |
| 200 |  |
| 400 |  |
| 800 |  |

- val loss and real-word rate after training 3x longer, at 128 **and** at 800 tokens:

**Character-level versus word-level (section 1.7):**

| | char | word |
|---|---|---|
| parameters |  |  |
| wall clock |  |  |
| real-word rate @128 tokens |  |  |
| real-word rate @800 tokens |  |  |

- vocabulary and coverage at min_count 1 / 2 / 3:
- `word_quality.py` real-word rate, 5 seeds x 5 lengths:
- your best word-level sample, pasted:

**Exercise 1.5, the GQA ablation** (`ablation_gqa.py --steps 500`):

| n_kv_heads | params | val loss | KV bytes @ 4096 tok | reduction | seconds/step |
|---|---|---|---|---|---|
| 6 (multi-head) |  |  |  |  |  |
| 2 (grouped) |  |  |  |  |  |
| 1 (multi-query) |  |  |  |  |  |

- what happened at `--lr 3e-2`, and what `grad_norm` did while it happened:
- loss ceiling with the `apply_rope` calls removed:

**Cached versus uncached generation** (`gen_timing.py`, note your `--threads`):

| generated tokens | uncached | cached | speedup |
|---|---|---|---|
| 64 |  |  |  |
| 128 |  |  |  |
| 256 |  |  |  |
| 512 |  |  |  |

**Optional, the TinyStories scale-up:** corpus size, vocabulary, parameters, steps,
final perplexity, and a sample:

## Week 2: architecture and deployment

- dense baseline params total / active, val loss (`--n-heads 6`):
- MoE (8 experts, top-2) params total / active, and the ratio:
- MoE val loss vs the dense baseline, and the wall-clock cost of the difference:
- auxiliary loss per layer at initialisation and at the end, with the config it
  belongs to:
- what routing collapse looked like with `--aux-loss-coef 0.0`:
- toy VLM: image tokens, text tokens, vision grad norm:
- your own model served: endpoint, latency per request:
- `vllm serve` flags that worked for `SmolLM2-135M-Instruct`:
- vLLM sequential (4 req) vs concurrent (16 req) tokens per second, and the ratio:
- ONNX export: file size, and max difference vs PyTorch:
- SmolVLM peak VRAM, and the `image_processor.size` longest_edge you needed:

| longest_edge | prompt tokens | peak VRAM | caption quality |
|---|---|---|---|
| 2048 |  |  |  |
| 1024 |  |  |  |
| 512 |  |  |  |

## Week 3: fine-tuning

- exact chat-template string, pasted from `00_template.py`:
- dataset source, size, audit output:
- `check_labels.py`: fraction of label tokens unmasked:
- LoRA config (`r`, `lora_alpha`, `target_modules`):
- effective batch size, seconds per step, peak VRAM:
- final train / eval loss, `mean_token_accuracy`:

**Paired before-and-after** (`evaluate.py`, greedy on both sides):

| metric | BASE | TUNED | delta |
|---|---|---|---|
| keyword_recall |  |  |  |
| keyword_precision |  |  |  |
| keyword_f1 |  |  |  |
| mean answer length |  |  |  |

Recall alone is length-biased: a model that learns to answer concisely scores worse
on it even when it improved. Record all four, and say which one you would defend.

**Full fine-tune versus LoRA (section 3.7):**

| config | trainable params | eval loss | seconds | peak GB |
|---|---|---|---|---|
| lora-r8 |  |  |  |  |
| lora-r32 |  |  |  |  |
| full |  |  |  |  |

- forgetting check: score of each model on 10 off-topic questions:
- failure modes seen in 20 hand-read outputs:
- GGUF size at Q4_K_M, and tokens/sec from `llama-bench`:
- Hub URL of the adapter:

## Week 4: agents

**Small-model tool calling (section 4.2)** (`measure_tools.py`, 20 questions):

| metric | SmolLM2-135M |
|---|---|
| format validity rate |  |
| tool selection accuracy |  |
| mean steps to answer |  |
| final answer accuracy |  |

- what broke when you deleted the worked example from the prompt:
- tools registered:
- wrong-tool cases, and the docstring fix that solved them:
- queries that hit the step budget:
- what the standard `tools` API did against this model, with
  `--enable-auto-tool-choice --tool-call-parser` set:
- MCP: tools and resources exposed, and a call that returned:
- deployed endpoint URL, and the container image size:

## Week 5: retrieval

- corpus: documents / chunks, chunk size and overlap:
- eval set: number of hand-labelled questions, and how you assigned gold chunks:

| Config | recall@5 | MRR | nDCG@5 |
|---|---|---|---|
| dense only |  |  |  |
| bm25 only |  |  |  |
| hybrid |  |  |  |
| hybrid + reranker |  |  |  |

- one query where hybrid beat dense, and why:
- one query where the reranker changed the top hit:
- whether your questions share exact wording with the passages, and which
  retriever that favours:
- generation with citations: did the model actually emit the markers:
- agent + retriever capstone: steps taken, tools fired, citations correct:
