# week2_arch/03_throughput.py
import time, asyncio
from openai import AsyncOpenAI

client = AsyncOpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
PROMPTS = [f"In one sentence, explain concept number {i} in distributed systems." for i in range(16)]

async def one(p):
    r = await client.chat.completions.create(
        model=MODEL, messages=[{"role": "user", "content": p}], max_tokens=80
    )
    return r.usage.completion_tokens

async def main():
    t0 = time.time()
    seq = 0
    for p in PROMPTS[:4]:
        seq += await one(p)
    t_seq = time.time() - t0
    print(f"sequential: 4 requests, {seq} tokens, {t_seq:.1f}s, {seq/t_seq:.1f} tok/s")

    t0 = time.time()
    results = await asyncio.gather(*[one(p) for p in PROMPTS])
    t_par = time.time() - t0
    total = sum(results)
    print(f"concurrent: 16 requests, {total} tokens, {t_par:.1f}s, {total/t_par:.1f} tok/s")

asyncio.run(main())
