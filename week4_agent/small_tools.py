"""Tool calling that works with a 135M model.

Small models cannot reliably emit the OpenAI `tool_calls` structure. They CAN
emit one line in a fixed format if you show them examples. This module parses
that line, validates it, and gives the model a correction when it gets it wrong.
"""
import re, json, math

TOOL_RE = re.compile(r"TOOL:\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\((.*?)\)\s*$", re.M | re.S)

def parse_tool_call(text, allowed):
    """Return (name, args_string) or None. Rejects unknown tools loudly."""
    m = TOOL_RE.search(text.strip())
    if not m:
        return None
    name, raw = m.group(1), m.group(2).strip()
    if name not in allowed:
        return ("__unknown__", name)
    return (name, raw)

def parse_args(raw, schema):
    """Parse `key=value, key=value` or a bare single positional value."""
    out = {}
    if not raw:
        return out
    if "=" not in raw:
        keys = list(schema)
        if len(keys) == 1:
            return {keys[0]: raw.strip().strip("'\"")}
        raise ValueError("expected key=value arguments")
    for part in re.split(r",(?=\s*[a-zA-Z_][a-zA-Z0-9_]*\s*=)", raw):
        k, _, v = part.partition("=")
        out[k.strip()] = v.strip().strip("'\"")
    missing = [k for k in schema if k not in out]
    if missing:
        raise ValueError(f"missing argument(s): {', '.join(missing)}")
    return out

def build_prompt(tools, question):
    """Few-shot the format. Small models copy formats far better than they follow rules."""
    lines = ["You can use these tools:"]
    for name, spec in tools.items():
        args = ", ".join(f"{k}=..." for k in spec["args"])
        lines.append(f"- TOOL: {name}({args})  # {spec['doc']}")
    lines += [
        "",
        "To use a tool, reply with exactly one line and nothing else:",
        "TOOL: name(arg=value)",
        "When you have the answer, reply with:",
        "ANSWER: your answer",
        "",
        "Example:",
        "Question: What is 6 times 7?",
        "TOOL: calculator(expression=6*7)",
        "Result: 42",
        "ANSWER: 42",
        "",
        f"Question: {question}",
    ]
    return "\n".join(lines)

TOOLS = {
    "calculator": {"args": ["expression"], "doc": "evaluate arithmetic",
                   "fn": lambda expression: str(eval(expression, {"__builtins__": {}},
                        {k: v for k, v in vars(math).items() if not k.startswith("_")}))},
    "word_count": {"args": ["text"], "doc": "count words",
                   "fn": lambda text: str(len(text.split()))},
}

def run_small_agent(complete, question, max_steps=4):
    """`complete(prompt) -> str` is any text-completion callable."""
    transcript = build_prompt(TOOLS, question)
    for _ in range(max_steps):
        out = complete(transcript).strip()
        if "ANSWER:" in out:
            return out.split("ANSWER:", 1)[1].strip(), transcript
        call = parse_tool_call(out, TOOLS)
        if call is None:
            transcript += f"\n{out}\nThat was not a valid line. Reply with exactly `TOOL: name(arg=value)` or `ANSWER: ...`."
            continue
        name, raw = call
        if name == "__unknown__":
            transcript += f"\n{out}\nUnknown tool `{raw}`. Available: {', '.join(TOOLS)}."
            continue
        try:
            args = parse_args(raw, TOOLS[name]["args"])
            result = TOOLS[name]["fn"](**args)
        except Exception as e:
            result = f"error: {type(e).__name__}: {e}"
        transcript += f"\nTOOL: {name}({raw})\nResult: {result}"
    return "stopped: step budget exhausted", transcript

if __name__ == "__main__":
    # well-formed call
    assert parse_tool_call("TOOL: calculator(expression=17*23+4)", TOOLS) == ("calculator", "expression=17*23+4")
    # bare positional
    assert parse_args("17*23+4", ["expression"]) == {"expression": "17*23+4"}
    # two named args
    assert parse_args("a=1, b=hello world", ["a", "b"]) == {"a": "1", "b": "hello world"}
    # value containing a comma must not split
    assert parse_args("text=one, two, three", ["text"]) == {"text": "one, two, three"}
    # missing arg
    try:
        parse_args("b=2", ["a", "b"]); raise SystemExit("should have failed")
    except ValueError as e:
        assert "missing" in str(e)
    # unknown tool
    assert parse_tool_call("TOOL: teleport(x=1)", TOOLS) == ("__unknown__", "teleport")
    # chatty preamble around the line, which is what small models actually do
    assert parse_tool_call("Sure, let me help!\nTOOL: word_count(text=a b c)", TOOLS) == ("word_count", "text=a b c")
    # no call at all
    assert parse_tool_call("I think the answer is 42.", TOOLS) is None
    print("parser tests: PASS")

    # full loop against a scripted small model, including a malformed first attempt
    script = iter([
        "Umm I will use the calc",                       # malformed
        "TOOL: calculator(expression=17*23+4)",
        "TOOL: word_count(text=the quick brown fox)",
        "ANSWER: 395 and 4 words",
    ])
    ans, tr = run_small_agent(lambda p: next(script), "17*23+4 and word count?")
    assert ans == "395 and 4 words", ans
    assert "Result: 395" in tr and "Result: 4" in tr
    assert "not a valid line" in tr
    print("loop with malformed recovery: PASS ->", ans)

    # error isolation and budget guard
    script2 = iter(["TOOL: calculator(expression=1/0)", "ANSWER: it failed"])
    a2, t2 = run_small_agent(lambda p: next(script2), "divide by zero")
    assert "ZeroDivisionError" in t2
    a3, _ = run_small_agent(lambda p: "TOOL: calculator(expression=1+1)", "loop", max_steps=3)
    assert "step budget" in a3
    print("error isolation + budget guard: PASS")
    print("ALL SMALL-MODEL TOOL TESTS PASSED")
