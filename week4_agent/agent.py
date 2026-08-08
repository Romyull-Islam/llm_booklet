# week4_agent/agent.py
import json
import os
from openai import OpenAI
from tools import TOOLS

# The Dockerfile sets LLM_BASE_URL=http://host.docker.internal:8000/v1 so the
# container can reach a model server on the host. Hardcoding localhost here made
# that env var dead: inside the container localhost:8000 is the container itself.
client = OpenAI(base_url=os.environ.get("LLM_BASE_URL", "http://localhost:8000/v1"),
                api_key="EMPTY")
MODEL = os.environ.get("LLM_MODEL", "HuggingFaceTB/SmolLM2-135M-Instruct")

SYSTEM = """You are a careful assistant with access to tools.

Rules:
- Use a tool whenever it would give a more reliable answer than your own recollection.
- Use the calculator for all arithmetic. Do not compute in your head.
- After receiving tool results, answer the user's question directly.
- If a tool errors twice, tell the user what failed instead of retrying forever."""

def run_agent(question, max_steps=6, system=SYSTEM, verbose=True, force_first_call=False):
    """force_first_call=True sends tool_choice="required" on the first turn only.

    Small models often *narrate* a tool call instead of making one: they reply
    "I will use the search tool" and then answer from memory. tool_choice="required"
    removes the choice for one turn, after which the loop goes back to "auto" so the
    model can still decide it is finished. Use it whenever the question cannot be
    answered without a tool, which is always true for retrieval.
    """
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": question}]
    schemas = [t["schema"] for t in TOOLS.values()]

    for step in range(max_steps):
        resp = client.chat.completions.create(
            model=MODEL, messages=messages, tools=schemas,
            tool_choice="required" if (force_first_call and step == 0) else "auto",
            temperature=0.0,           # deterministic: agents should not be creative about control flow
        )
        msg = resp.choices[0].message

        messages.append({
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [
                {"id": c.id, "type": "function",
                 "function": {"name": c.function.name, "arguments": c.function.arguments}}
                for c in (msg.tool_calls or [])
            ] or None,
        })

        if not msg.tool_calls:
            return msg.content, messages

        for call in msg.tool_calls:
            name = call.function.name
            try:
                args = json.loads(call.function.arguments or "{}")
                result = TOOLS[name]["fn"](**args) if name in TOOLS else f"unknown tool: {name}"
            except Exception as e:
                result = f"tool error: {type(e).__name__}: {e}"
            if verbose:
                print(f"  [step {step}] {name}({call.function.arguments}) -> {str(result)[:120]}")
            messages.append({"role": "tool", "tool_call_id": call.id, "content": str(result)[:4000]})

    return "stopped: step budget exhausted", messages

if __name__ == "__main__":
    ans, _ = run_agent("What is 17 * 23 + 4, and how many words are in 'the quick brown fox'?")
    print("\nANSWER:", ans)
