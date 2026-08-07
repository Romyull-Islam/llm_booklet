# Course notebook

Record every measurement here. These become your CV bullets and, if you go the
academia route, your methods section.

## Week 1: build and train

- machine (CPU or GPU), cores or card:
- parameter count reported by `train.py`:
- `test_cache.py` max logit difference (want < 1e-4):
- first training loss vs `ln(vocab_size)`:
- final train / val loss, wall clock:
- seconds per step:
- sample output at the end, pasted:
- real-word rate from `sample_quality.py`:

**Quality versus generation length (run 3 seeds each):**

| generated tokens | real-word rate |
|---|---|
| 100 |  |
| 128 (trained context) |  |
| 200 |  |
| 400 |  |
| 800 |  |

- val loss and real-word rate after training 3x longer:

**Character-level versus word-level (section 1.7):**

| | char | word |
|---|---|---|
| parameters |  |  |
| wall clock |  |  |
| real-word rate @128 tokens |  |  |
| real-word rate @800 tokens |  |  |

- vocabulary and coverage at min_count 1 / 2 / 3:
- your best word-level sample, pasted:

**Exercise 1.5, the GQA ablation:**

| n_kv_heads | final val loss | KV bytes @ 4096 tok | seconds/step |
|---|---|---|---|
| 6 (multi-head) |  |  |  |
| 2 (grouped) |  |  |  |
| 1 (multi-query) |  |  |  |

- what happened at `--lr 3e-2`:
- loss ceiling with RoPE removed:
- cached vs uncached generation time, and the ratio:

## Week 2: architecture and deployment

- dense params total / active:
- MoE (8 experts, top-2) params total / active, and the ratio:
- MoE final val loss vs dense at equal active params:
- auxiliary loss per layer at end of training:
- what routing collapse looked like with `aux_loss_coef=0.0`:
- toy VLM: image tokens, text tokens, vision grad norm:
- your model served: endpoint, latency per request:
- vLLM flags that worked for `Qwen3-4B-Instruct-2507`:
- vLLM sequential (4 req) vs concurrent (16 req) tokens per second:
- VLM peak VRAM and the `max_pixels` needed to avoid OOM:

## Week 3: fine-tuning

- exact chat-template string, pasted from `00_template.py`:
- dataset source, size, audit output:
- LoRA config (`r`, `lora_alpha`, `target_modules`):
- effective batch size, seconds per step, peak VRAM:
- final train / eval loss, `mean_token_accuracy`:
- BASE score on held-out set:
- TUNED score on held-out set:
- delta:

**Full fine-tune versus LoRA (section 3.7):**

| config | trainable params | eval loss | seconds | peak GB |
|---|---|---|---|---|
| lora-r8 |  |  |  |  |
| lora-r32 |  |  |  |  |
| full |  |  |  |  |

- forgetting check: score of each model on 10 off-topic questions:
- failure modes seen in 20 hand-read outputs:
- GGUF size at Q4_K_M, Hub URL of the adapter:

## Week 4: agents

**Small-model tool calling (section 4.2), over 20 questions:**

| metric | SmolLM2-135M |
|---|---|
| format validity rate |  |
| tool selection accuracy |  |
| mean steps to answer |  |
| final answer accuracy |  |

- what broke when you deleted the worked example from the prompt:
- tools registered:
- average steps per query:
- wrong-tool cases, and the docstring fix that solved them:
- queries that hit the step budget:
- deployed endpoint URL:

## Week 5: retrieval

- corpus: documents / chunks, chunk size and overlap:
- eval set: number of hand-labelled questions:

| Config | recall@5 | MRR | nDCG@5 |
|---|---|---|---|
| dense only |  |  |  |
| bm25 only |  |  |  |
| hybrid |  |  |  |
| hybrid + reranker |  |  |  |

- one query where hybrid beat dense, and why:
- one query where the reranker changed the top hit:
- agent + retriever capstone: steps taken, citations correct:
