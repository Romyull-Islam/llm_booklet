"""
serve_smollm.py: an OpenAI-compatible endpoint for SmolLM2-135M-Instruct.

WHY THIS EXISTS. The booklet serves this model with `vllm serve`. vLLM detects
its platform exclusively through NVML, and this machine has an NVML/RM version
mismatch (kernel module 580.159.03, userspace libnvidia-ml 580.173.02), so vLLM
exits with "Failed to infer device type" before it ever touches the GPU. torch
is unaffected because it uses the CUDA driver API. A reboot loads the matching
module and `vllm serve` works; until then this stands in on the same port and
the same /v1 path, so 02_client.py, 03_throughput.py and 05_ui.py run unchanged.

WHAT IT IS NOT. This does naive fixed-window batching, not vLLM's continuous
batching with PagedAttention. It will show a concurrency win, but a smaller one
than vLLM would. Do not record its number as a vLLM number.
"""
import asyncio
import json
import time
import uuid

import torch
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
BATCH_WINDOW_S = 0.010          # how long to wait for more requests before running
MAX_BATCH = 32

app = FastAPI(title="SmolLM2 (transformers stand-in for vLLM)")

tokenizer = AutoTokenizer.from_pretrained(MODEL)
tokenizer.padding_side = "left"                    # decoder-only batched generation
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).cuda().eval()

QUEUE: asyncio.Queue = asyncio.Queue()


class Msg(BaseModel):
    role: str
    content: str | list


class Req(BaseModel):
    model: str = MODEL
    messages: list[Msg]
    temperature: float = 0.7
    max_tokens: int = 300
    stream: bool = False


def render(messages):
    return tokenizer.apply_chat_template(
        [{"role": m.role, "content": m.content} for m in messages],
        tokenize=False, add_generation_prompt=True,
    )


@torch.inference_mode()
def run_batch(prompts, max_tokens, temperature):
    enc = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
    out = model.generate(
        **enc,
        max_new_tokens=max_tokens,
        do_sample=temperature > 0,
        temperature=temperature if temperature > 0 else None,
        pad_token_id=tokenizer.pad_token_id,
    )
    gen = out[:, enc["input_ids"].shape[-1]:]
    texts = tokenizer.batch_decode(gen, skip_special_tokens=True)
    n_prompt = int(enc["attention_mask"].sum(dim=1).sum().item())
    counts = [int((g != tokenizer.pad_token_id).sum().item()) for g in gen]
    return texts, counts, n_prompt


async def batcher():
    """Collect requests arriving within BATCH_WINDOW_S and run them as one batch."""
    while True:
        item = await QUEUE.get()
        batch = [item]
        deadline = time.perf_counter() + BATCH_WINDOW_S
        while len(batch) < MAX_BATCH:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                break
            try:
                batch.append(await asyncio.wait_for(QUEUE.get(), timeout=remaining))
            except asyncio.TimeoutError:
                break
        # One generate() call for the whole group. Uniform settings per batch.
        max_tokens = max(b["max_tokens"] for b in batch)
        temperature = batch[0]["temperature"]
        try:
            texts, counts, n_prompt = await asyncio.to_thread(
                run_batch, [b["prompt"] for b in batch], max_tokens, temperature)
            for b, t, c in zip(batch, texts, counts):
                b["future"].set_result((t, c, n_prompt // len(batch)))
        except Exception as exc:                       # noqa: BLE001
            for b in batch:
                b["future"].set_exception(exc)


@app.on_event("startup")
async def _startup():
    asyncio.create_task(batcher())


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL}


@app.get("/v1/models")
def models():
    return {"object": "list", "data": [{"id": MODEL, "object": "model"}]}


@app.post("/v1/chat/completions")
async def chat(r: Req):
    prompt = render(r.messages)

    if r.stream:
        # Streaming bypasses the batcher: one request, token by token.
        from threading import Thread
        from transformers import TextIteratorStreamer

        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True,
                                        skip_special_tokens=True)
        enc = tokenizer(prompt, return_tensors="pt").to(model.device)
        kwargs = dict(**enc, max_new_tokens=r.max_tokens, streamer=streamer,
                      do_sample=r.temperature > 0,
                      temperature=r.temperature if r.temperature > 0 else None,
                      pad_token_id=tokenizer.pad_token_id)
        Thread(target=model.generate, kwargs=kwargs, daemon=True).start()

        def sse():
            cid = f"chatcmpl-{uuid.uuid4().hex[:12]}"
            for piece in streamer:
                if not piece:
                    continue
                yield "data: " + json.dumps({
                    "id": cid, "object": "chat.completion.chunk", "model": r.model,
                    "choices": [{"index": 0, "delta": {"content": piece},
                                 "finish_reason": None}],
                }) + "\n\n"
            yield "data: " + json.dumps({
                "id": cid, "object": "chat.completion.chunk", "model": r.model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }) + "\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(sse(), media_type="text/event-stream")

    fut = asyncio.get_running_loop().create_future()
    await QUEUE.put({"prompt": prompt, "max_tokens": r.max_tokens,
                     "temperature": r.temperature, "future": fut})
    text, n_completion, n_prompt = await fut
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "model": r.model,
        "choices": [{"index": 0, "finish_reason": "stop",
                     "message": {"role": "assistant", "content": text}}],
        "usage": {"prompt_tokens": n_prompt, "completion_tokens": n_completion,
                  "total_tokens": n_prompt + n_completion},
    }
