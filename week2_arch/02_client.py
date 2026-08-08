# week2_arch/02_client.py
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"

# Plain chat
resp = client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": "Give me three bullet points on why PagedAttention helps throughput."}],
    temperature=0.7,
    max_tokens=300,
)
print(resp.choices[0].message.content)
print("usage:", resp.usage)

# Streaming, which is what any real UI needs
stream = client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": "Count from 1 to 20, comma separated."}],
    stream=True,
)
for chunk in stream:
    delta = chunk.choices[0].delta.content
    if delta:
        print(delta, end="", flush=True)
print()
