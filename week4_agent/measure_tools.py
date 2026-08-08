"""
measure_tools.py: section 4.2 "Measure it".

The booklet says "Write 20 questions where you know which tool should be called,
and score: format validity rate, tool selection accuracy, steps to answer, final
answer accuracy" and gives the table headings, but ships no question set and no
scorer. This is that.

Runs against whatever OpenAI-compatible endpoint is on LLM_BASE_URL.
"""
import os
import re
import statistics

from openai import OpenAI

from small_tools import TOOLS, build_prompt, parse_tool_call, parse_args

client = OpenAI(base_url=os.environ.get("LLM_BASE_URL", "http://localhost:8000/v1"),
                api_key="EMPTY")
MODEL = os.environ.get("LLM_MODEL", "HuggingFaceTB/SmolLM2-135M-Instruct")

# (question, expected tool, expected answer)
QUESTIONS = [
    ("What is 17 * 23 + 4?",                              "calculator", "395"),
    ("What is 144 divided by 12?",                        "calculator", "12"),
    ("What is 2 to the power of 10?",                     "calculator", "1024"),
    ("Compute 3 * (2 + 1).",                              "calculator", "9"),
    ("What is 1000 - 447?",                               "calculator", "553"),
    ("What is 25 * 4?",                                   "calculator", "100"),
    ("Compute 99 + 1.",                                   "calculator", "100"),
    ("What is 7 * 8 - 6?",                                "calculator", "50"),
    ("What is 81 divided by 9?",                          "calculator", "9"),
    ("Compute 15 + 27 + 3.",                              "calculator", "45"),
    ("How many words are in: the quick brown fox",        "word_count", "4"),
    ("Count the words in: hello world",                   "word_count", "2"),
    ("How many words in: one two three four five",        "word_count", "5"),
    ("Count the words in: a",                             "word_count", "1"),
    ("How many words are in: to be or not to be",         "word_count", "6"),
    ("Count the words in: the rain in spain",             "word_count", "4"),
    ("How many words in: I think therefore I am",         "word_count", "5"),
    ("Count the words in: red green blue",                "word_count", "3"),
    ("How many words are in: this sentence has five words", "word_count", "5"),
    ("Count the words in: yes no maybe so",               "word_count", "4"),
]


def complete(prompt):
    r = client.chat.completions.create(
        model=MODEL, messages=[{"role": "user", "content": prompt}],
        temperature=0.0, max_tokens=64,
    )
    return r.choices[0].message.content or ""


def run_one(question):
    """Instrumented copy of run_small_agent that records what happened."""
    transcript = build_prompt(TOOLS, question)
    first_parsed, chosen_tool, steps = None, None, 0
    for _ in range(4):
        out = complete(transcript).strip()
        steps += 1
        call = parse_tool_call(out, TOOLS)
        if first_parsed is None:
            first_parsed = bool(call) and call[0] != "__unknown__" or "ANSWER:" in out
        if "ANSWER:" in out:
            return out.split("ANSWER:", 1)[1].strip(), first_parsed, chosen_tool, steps
        if call is None:
            transcript += (f"\n{out}\nThat was not a valid line. Reply with exactly "
                           "`TOOL: name(arg=value)` or `ANSWER: ...`.")
            continue
        name, raw = call
        if name == "__unknown__":
            transcript += f"\n{out}\nUnknown tool `{raw}`. Available: {', '.join(TOOLS)}."
            continue
        if chosen_tool is None:
            chosen_tool = name
        try:
            result = TOOLS[name]["fn"](**parse_args(raw, TOOLS[name]["args"]))
        except Exception as e:                              # noqa: BLE001
            result = f"error: {type(e).__name__}: {e}"
        transcript += f"\nTOOL: {name}({raw})\nResult: {result}"
    return "stopped: step budget exhausted", first_parsed, chosen_tool, steps


valid, correct_tool, correct_answer, all_steps = [], [], [], []
print(f"{'question':<48}{'tool':>12}{'steps':>7}  answer")
for q, want_tool, want_ans in QUESTIONS:
    ans, parsed, tool, steps = run_one(q)
    valid.append(bool(parsed))
    correct_tool.append(tool == want_tool)
    # Accept the right number appearing anywhere in the final answer.
    correct_answer.append(bool(re.search(rf"(?<!\d){re.escape(want_ans)}(?!\d)", ans)))
    all_steps.append(steps)
    print(f"{q[:46]:<48}{str(tool):>12}{steps:>7}  {ans[:40]}")

n = len(QUESTIONS)
print()
print(f"{'metric':<28}{'SmolLM2-135M':>14}")
print(f"{'format validity rate':<28}{sum(valid)/n:>13.0%}")
print(f"{'tool selection accuracy':<28}{sum(correct_tool)/n:>13.0%}")
print(f"{'mean steps to answer':<28}{statistics.mean(all_steps):>13.2f}")
print(f"{'final answer accuracy':<28}{sum(correct_answer)/n:>13.0%}")
