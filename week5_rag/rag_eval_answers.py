"""
rag_eval_answers.py: does the model answer correctly once retrieval has done its job?

Two paths are worth separating, because they fail for different reasons:

  plain RAG      retrieve -> stuff the passages into the prompt -> answer.
                 The model never decides anything. Tests reading comprehension.

  agentic RAG    the model decides to search, writes the query, reads the result,
                 and decides whether it can answer. Tests tool use as well.

Scoring, and why substring matching is not good enough. The obvious metric is
"does the answer contain the anchor phrase". It is wrong in the same way
keyword_recall was wrong in section 3.5: it punishes a correct paraphrase. Asked
why RoPE rotates, a model that answers "it rotates q and k by an angle
proportional to position, so the score depends on their separation" is right, and
a substring test for "depends only on (m - n)" scores it zero.

So an answer counts as correct if EITHER the anchor appears verbatim, OR the
answer is semantically close to the passage that answers the question, measured
with the same MiniLM encoder the retriever uses. Both are reported, so you can see
how much the strict version undercounts.

    python week5_rag/rag_eval_answers.py --mode plain   --n 12
    python week5_rag/rag_eval_answers.py --mode agentic --n 12
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "week5_rag"))
sys.path.insert(0, str(ROOT / "week4_agent"))

ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["plain", "agentic"], default="plain")
ap.add_argument("--n", type=int, default=12)
ap.add_argument("--top-k", type=int, default=4)
a = ap.parse_args()

MODEL = os.environ.get("LLM_MODEL", "HuggingFaceTB/SmolLM2-135M-Instruct")


def norm(s):
    return re.sub(r"[^a-z0-9 ]", " ", s.lower())


records = json.load(open(ROOT / "data" / "chunks.json"))
questions = json.load(open(ROOT / "data" / "eval_questions.json"))[: a.n]

from retriever import HybridRetriever                                   # noqa: E402
retriever = HybridRetriever(records)

if a.mode == "plain":
    from generate import answer as rag_answer

    def run(q):
        out = rag_answer(q, retriever, top_k=a.top_k)
        return out["answer"], 1
else:
    import rag_agent as RA                                              # noqa: E402
    from agent import run_agent                                         # noqa: E402

    def run(q):
        ans, trace = run_agent(q, system=RA.SYSTEM, max_steps=4,
                               verbose=False, force_first_call=True)
        return ans, sum(1 for m in trace if m.get("role") == "tool")


# Did retrieval even surface the answer? Separating this from the model's job is the
# whole point: a wrong answer over a correct passage is a different bug from a wrong
# answer over a missed passage.
SIM_THRESHOLD = 0.55        # cosine on MiniLM embeddings; see the note above

JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "")


def judge(question, gold_passage, answer):
    """LLM-as-judge. Returns True/False, or None if no judge is configured.

    Judging free text is the honest way to score this, and it needs a model at
    least as capable as the one being judged. Point JUDGE_MODEL at a second
    endpoint (JUDGE_BASE_URL) if you have one; self-judging is better than
    substring matching but grades generously.
    """
    if not JUDGE_MODEL:
        return None
    from openai import OpenAI
    jc = OpenAI(base_url=os.environ.get("JUDGE_BASE_URL",
                                        os.environ.get("LLM_BASE_URL",
                                                       "http://localhost:8000/v1")),
                api_key="EMPTY")
    prompt = (
        "You are grading one answer. Reply with exactly one word: CORRECT or WRONG.\n\n"
        f"QUESTION: {question}\n\n"
        f"REFERENCE PASSAGE (the ground truth):\n{gold_passage[:1200]}\n\n"
        f"ANSWER TO GRADE:\n{answer[:800]}\n\n"
        "The answer is CORRECT if it states what the reference passage says, in any "
        "wording. It is WRONG if it contradicts the passage, invents specifics not in "
        "it, or does not address the question. Reply CORRECT or WRONG."
    )
    r = jc.chat.completions.create(model=JUDGE_MODEL, temperature=0.0, max_tokens=4,
                                   messages=[{"role": "user", "content": prompt}])
    return "CORRECT" in (r.choices[0].message.content or "").upper()

hits = strict = lenient = 0
steps_total = 0
print(f"model {MODEL}   mode {a.mode}   {len(questions)} questions\n")
for ex in questions:
    q, anchor = ex["question"], ex["anchor"]
    retrieved = retriever.retrieve(q, top_k=a.top_k)
    # Match on chunk id, not on the anchor string. A different chunk can state the
    # same fact in different words, and an anchor test would call that a miss.
    got = {r["chunk_id"] for r in retrieved}
    in_context = bool(got & set(ex["gold_chunk_ids"]))
    hits += in_context
    ans, steps = run(q)
    steps_total += steps

    exact = norm(anchor)[:40].strip() in norm(ans)
    # Semantic check against the gold passage, not against the anchor alone: the
    # anchor is a locator, the passage is what a correct answer paraphrases.
    gold_text = next((r["text"] for r in records
                      if r["chunk_id"] in ex["gold_chunk_ids"]), anchor)
    emb = retriever.vec.model.encode([ans, gold_text], normalize_embeddings=True)
    sim = float(emb[0] @ emb[1])
    ok = exact or sim >= SIM_THRESHOLD

    verdict = judge(q, gold_text, ans)
    if verdict is not None:
        ok = verdict

    strict += exact
    lenient += ok
    mark = "ANS" if ok else "   "
    how = ("judge" if verdict is not None else
           ("exact" if exact else f"sim {sim:.2f}"))
    print(f"  [{'RET' if in_context else '   '}][{mark}] {q[:52]}  ({how})")
    print(f"        -> {ans.strip()[:104]}")

n = len(questions)
print(f"\n  retrieval put the answer in context : {hits}/{n}  ({hits/n:.0%})")
print(f"  answered, exact phrase match        : {strict}/{n}  ({strict/n:.0%})")
label = "judged correct" if JUDGE_MODEL else "exact or semantic match"
print(f"  answered, {label:<25} : {lenient}/{n}  ({lenient/n:.0%})")
print(f"  mean tool calls                     : {steps_total/n:.2f}")
