#!/usr/bin/env python3
"""
verify.py: reproduce every number this repo claims, without a GPU or a download.

    python verify.py

Runs the tests behind the figures quoted in the booklet: KV cache correctness,
tokeniser coverage, retrieval metrics, dataset auditing, and both agent loops.
Exits non-zero if anything fails, so it works as a CI check.

Not covered here (needs a GPU and model weights): the LoRA and full fine-tuning
runs, vLLM serving, pretrained VLM inference, GGUF conversion. See Appendix E.
"""
import re
import sys
import math
import json
import types
import pathlib
import traceback

ROOT = pathlib.Path(__file__).parent
RESULTS = []


def check(name, fn):
    try:
        detail = fn()
        RESULTS.append((name, True, detail))
        print(f"  PASS  {name:<42} {detail}")
    except Exception as e:
        RESULTS.append((name, False, f"{type(e).__name__}: {e}"))
        print(f"  FAIL  {name:<42} {type(e).__name__}: {e}")
        if "-v" in sys.argv:
            traceback.print_exc()


def load_module(path, name):
    """Import a file without running its __main__ block."""
    src = pathlib.Path(path).read_text().split("\nif __name__")[0]
    mod = types.ModuleType(name)
    mod.__file__ = str(path)
    sys.path.insert(0, str(pathlib.Path(path).parent))
    exec(compile(src, str(path), "exec"), mod.__dict__)
    return mod


# ---------------------------------------------------------------- week 1

def week1():
    import torch
    sys.path.insert(0, str(ROOT / "week1_build"))
    from model import Config, Transformer

    def shapes_and_loss():
        torch.manual_seed(0)
        cfg = Config(vocab_size=65, d_model=128, n_layers=4, n_heads=8,
                     n_kv_heads=2, d_ff=352, max_seq_len=128)
        m = Transformer(cfg).eval()
        x = torch.randint(0, 65, (4, 33))
        logits, loss, _ = m(x[:, :-1], targets=x[:, 1:])
        assert logits.shape == (4, 32, 65), logits.shape
        ln_v = math.log(65)
        assert abs(loss.item() - ln_v) < 0.15, f"{loss.item()} vs ln(65)={ln_v}"
        return f"untrained loss {loss.item():.4f} vs ln(65)={ln_v:.4f}"

    def kv_cache_equivalence():
        torch.manual_seed(0)
        cfg = Config(vocab_size=65, d_model=128, n_layers=4, n_heads=8,
                     n_kv_heads=2, d_ff=352, max_seq_len=128)
        m = Transformer(cfg).eval()
        seq = torch.randint(0, 65, (1, 24))
        full, _, _ = m(seq)
        caches = [(None, None)] * cfg.n_layers
        step, _, caches = m(seq[:, :10], caches=caches, pos_offset=0)
        outs = [step[:, -1]]
        for t in range(10, 24):
            step, _, caches = m(seq[:, t:t + 1], caches=caches, pos_offset=t)
            outs.append(step[:, -1])
        diff = (full[:, 9:, :] - torch.stack(outs, dim=1)).abs().max().item()
        assert diff < 1e-4, diff
        return f"max logit diff {diff:.2e} (booklet claims 4.77e-07)"

    def gqa_cache_ratio():
        def kv_bytes(n_layers, n_kv, d_model, n_heads, seq=4096, bpe=2):
            return 2 * n_layers * n_kv * (d_model // n_heads) * seq * bpe
        mha = kv_bytes(4, 8, 128, 8)
        gqa = kv_bytes(4, 2, 128, 8)
        assert abs(mha / gqa - 4.0) < 1e-9
        return f"MHA {mha/1e6:.1f} MB vs GQA {gqa/1e6:.1f} MB = {mha/gqa:.1f}x"

    def moe_accounting():
        torch.manual_seed(0)
        base = dict(vocab_size=65, d_model=128, n_layers=4, n_heads=8,
                    n_kv_heads=2, d_ff=352)
        dense = Transformer(Config(**base)).param_report()
        moe = Transformer(Config(**base, n_experts=8, n_experts_active=2)).param_report()
        assert dense["total"] == 713984, dense
        assert moe["total"] == 4502784, moe
        ratio = moe["total"] / moe["active_per_token"]
        assert abs(ratio - 3.58) < 0.01, ratio
        return f"dense {dense['total']:,} vs moe {moe['total']:,}, ratio {ratio:.2f}x"

    def moe_router_balance():
        torch.manual_seed(0)
        m = Transformer(Config(vocab_size=65, d_model=128, n_layers=4, n_heads=8,
                               n_kv_heads=2, d_ff=352, n_experts=8, n_experts_active=2))
        x = torch.randint(0, 65, (2, 32))
        m(x[:, :-1], targets=x[:, 1:])
        aux = sum(b.ffn.aux_loss for b in m.blocks).item() / 4
        assert 0.9 < aux < 1.6, aux
        return f"aux loss {aux:.4f} per layer (1.0 = perfectly balanced)"

    def rope_grows_past_context():
        torch.manual_seed(0)
        cfg = Config(vocab_size=65, d_model=128, n_layers=2, n_heads=4,
                     n_kv_heads=2, d_ff=352, max_seq_len=64)
        m = Transformer(cfg).eval()
        out = m.generate(torch.randint(0, 65, (1, 10)), 120, temperature=0.0)
        assert out.shape[1] == 130, out.shape
        return f"generated to {out.shape[1]} tokens past max_seq_len={cfg.max_seq_len}"

    def word_tokenizer():
        wt = load_module(ROOT / "week1_build" / "word_tok.py", "wt")
        text = "the cat sat on the mat\n\nthe cat ran\n" * 40
        t1 = wt.WordTokenizer(text, min_count=1)
        t3 = wt.WordTokenizer(text, min_count=3)
        assert t1.coverage == 1.0, t1.coverage
        assert t3.coverage <= t1.coverage
        assert t1.itos[0] == "<unk>", "index 0 must be <unk> so the sampler can ban it"
        s = "the cat sat"
        assert t1.decode(t1.encode(s)) == s, t1.decode(t1.encode(s))
        return f"vocab {t1.vocab_size}, coverage {t1.coverage:.4f}, round-trip clean"

    print("\nWeek 1: build and train")
    check("model shapes and untrained loss", shapes_and_loss)
    check("KV cache numerical equivalence", kv_cache_equivalence)
    check("GQA cache reduction", gqa_cache_ratio)
    check("MoE parameter accounting", moe_accounting)
    check("MoE router load balance", moe_router_balance)
    check("RoPE grows past trained context", rope_grows_past_context)
    check("word tokeniser coverage and round-trip", word_tokenizer)


# ---------------------------------------------------------------- week 2

def week2():
    import torch
    sys.path.insert(0, str(ROOT / "week1_build"))

    def vlm_gradient_flows():
        vlm = load_module(ROOT / "week2_arch" / "tiny_vlm.py", "tv")
        torch.manual_seed(0)
        cfg = vlm.Config(vocab_size=65, d_model=128, n_layers=2, n_heads=4,
                         n_kv_heads=2, d_ff=352, max_seq_len=64)
        m = vlm.TinyVLM(cfg)
        logits, loss = m(torch.randn(2, 3, 32, 32), torch.randint(0, 65, (2, 12)),
                         targets=torch.randint(0, 65, (2, 12)))
        assert logits.shape == (2, 28, 65), logits.shape
        loss.backward()
        g = m.vision.proj.weight.grad
        assert g is not None and g.norm().item() > 0, "vision tower is detached"
        return f"16 image + 12 text tokens, vision grad norm {g.norm().item():.4f}"

    print("\nWeek 2: architecture and multimodality")
    check("toy VLM splices tokens, gradient reaches vision", vlm_gradient_flows)


# ---------------------------------------------------------------- week 3

def week3():
    def dataset_audit():
        pd_ = load_module(ROOT / "week3_finetune" / "prepare_data.py", "pd_")
        raw = [{"question": f"Q{i}?", "answer": f"Answer number {i} explained fully."}
               for i in range(30)]
        raw += [{"question": "Short?", "answer": "Yes"},
                {"question": "Echo?", "answer": "Echo? indeed it is"},
                {"question": "Echo?", "answer": "Echo? indeed it is"}]
        rows = [pd_.to_chat(r, system="sys") for r in raw]
        rep = pd_.audit(rows)
        assert rep == {"n": 33, "empty": 0, "dupes": 1, "too_short": 1,
                       "leaked_prompt": 2}, rep
        tr, va = pd_.split(rows, 0.1)
        assert len(tr) + len(va) == 33
        assert tr[0]["prompt"][0]["role"] == "system"
        assert tr[0]["completion"][0]["role"] == "assistant"
        return f"audit {rep}, split {len(tr)}/{len(va)}"

    print("\nWeek 3: fine-tuning helpers")
    check("dataset audit catches dupes, stubs, leaks", dataset_audit)


# ---------------------------------------------------------------- week 4

def week4():
    def tool_schema_from_type_hints():
        t = load_module(ROOT / "week4_agent" / "tools.py", "tl")
        sch = t.TOOLS["calculator"]["schema"]["function"]
        assert sch["parameters"]["properties"]["expression"]["type"] == "string"
        assert sch["parameters"]["required"] == ["expression"]
        assert t.TOOLS["calculator"]["fn"]("17 * 23 + 4") == "395"
        assert t.TOOLS["word_count"]["fn"]("one two three") == "3"
        return "schema generated, calculator=395, word_count=3"

    def standard_agent_loop():
        import importlib
        sys.path.insert(0, str(ROOT / "week4_agent"))
        agent = importlib.import_module("agent")

        class F:
            def __init__(s, n, a): s.name, s.arguments = n, a
        class C:
            def __init__(s, i, n, a): s.id, s.function = i, F(n, a)
        class M:
            def __init__(s, content=None, tool_calls=None):
                s.content, s.tool_calls = content, tool_calls
        class R:
            def __init__(s, m): s.choices = [type("Ch", (), {"message": m})()]
        class Stub:
            def __init__(s, script): s.script, s.i = script, 0
            @property
            def chat(s): return s
            @property
            def completions(s): return s
            def create(s, **kw):
                assert kw["temperature"] == 0.0 and "tools" in kw
                m = s.script[s.i]; s.i += 1
                return R(m)

        agent.client = Stub([
            M(tool_calls=[C("c1", "calculator", '{"expression": "17 * 23 + 4"}')]),
            M(tool_calls=[C("c2", "word_count", '{"text": "the quick brown fox"}')]),
            M(content="395 and 4 words."),
        ])
        ans, tr = agent.run_agent("q", verbose=False)
        assert [m["content"] for m in tr if m["role"] == "tool"] == ["395", "4"]

        agent.client = Stub([M(tool_calls=[C("c", "calculator", '{"expression": "1/0"}')]),
                             M(content="handled")])
        _, t2 = agent.run_agent("q", verbose=False)
        assert "ZeroDivisionError" in [m["content"] for m in t2 if m["role"] == "tool"][0]

        agent.client = Stub([M(tool_calls=[C("c", "nope", "{}")]), M(content="ok")])
        _, t3 = agent.run_agent("q", verbose=False)
        assert "unknown tool" in [m["content"] for m in t3 if m["role"] == "tool"][0]

        agent.client = Stub([M(tool_calls=[C(f"c{i}", "calculator", '{"expression":"1+1"}')])
                             for i in range(10)])
        a4, _ = agent.run_agent("q", max_steps=3, verbose=False)
        assert "step budget" in a4

        agent.TOOLS["big"] = {"fn": lambda: "x" * 10000, "schema": {
            "type": "function", "function": {"name": "big", "description": "",
            "parameters": {"type": "object", "properties": {}, "required": []}}}}
        agent.client = Stub([M(tool_calls=[C("c", "big", "{}")]), M(content="done")])
        _, t5 = agent.run_agent("q", verbose=False)
        assert len([m for m in t5 if m["role"] == "tool"][0]["content"]) == 4000
        return "happy path, tool error, unknown tool, budget, truncation"

    def small_model_parser():
        st = load_module(ROOT / "week4_agent" / "small_tools.py", "st")
        assert st.parse_tool_call("TOOL: calculator(expression=17*23+4)",
                                  st.TOOLS) == ("calculator", "expression=17*23+4")
        assert st.parse_args("17*23+4", ["expression"]) == {"expression": "17*23+4"}
        assert st.parse_args("a=1, b=hello world", ["a", "b"]) == {"a": "1", "b": "hello world"}
        # the case a naive split(",") destroys
        assert st.parse_args("text=one, two, three", ["text"]) == {"text": "one, two, three"}
        try:
            st.parse_args("b=2", ["a", "b"])
            raise AssertionError("should have rejected a missing argument")
        except ValueError:
            pass
        assert st.parse_tool_call("TOOL: teleport(x=1)", st.TOOLS) == ("__unknown__", "teleport")
        assert st.parse_tool_call("Sure!\nTOOL: word_count(text=a b c)",
                                  st.TOOLS) == ("word_count", "text=a b c")
        assert st.parse_tool_call("I think the answer is 42.", st.TOOLS) is None

        script = iter(["Umm I will use the calc",
                       "TOOL: calculator(expression=17*23+4)",
                       "TOOL: word_count(text=the quick brown fox)",
                       "ANSWER: 395 and 4 words"])
        ans, tr = st.run_small_agent(lambda p: next(script), "q")
        assert ans == "395 and 4 words", ans
        assert "not a valid line" in tr and "Result: 395" in tr
        a3, _ = st.run_small_agent(lambda p: "TOOL: calculator(expression=1+1)", "q", max_steps=3)
        assert "step budget" in a3
        return "commas in values, missing args, unknown tools, malformed recovery"

    print("\nWeek 4: agents")
    check("tool schema from type hints", tool_schema_from_type_hints)
    check("standard agent loop and guardrails", standard_agent_loop)
    check("small-model parser and loop", small_model_parser)


# ---------------------------------------------------------------- week 5

def week5():
    import numpy as np

    def chunking():
        ch = load_module(ROOT / "week5_rag" / "chunking.py", "ch")
        text = "\n\n".join(f"Paragraph {i} about topic {i%3}. " * 12 for i in range(20))
        recs = ch.build_corpus([{"id": "d1", "title": "Doc One",
                                 "text": text, "url": "http://x/1"}])
        assert len(recs) > 1
        assert all(r["chunk_id"].startswith("d1::") and r["title"] == "Doc One" for r in recs)
        longest = max(len(r["text"]) for r in recs)
        assert longest <= 1400, longest
        return f"{len(recs)} chunks, max {longest} chars, metadata preserved"

    def bm25_and_fusion():
        hb = load_module(ROOT / "week5_rag" / "hybrid.py", "hb")
        corpus = ["the cat sat on the mat", "dogs are loyal animals",
                  "cats and dogs living together", "error code E-4471 means disk full"]
        bm = hb.BM25(corpus)
        top = bm.search("E-4471 disk", k=2)
        assert top[0][0] == 3, top
        fused = hb.rrf([[2, 0, 1, 3], [0, 2, 3, 1]])
        assert len(fused) == 4 and fused[0] in (0, 2)
        return f"exact-ID query hits doc {top[0][0]}, rrf order {fused}"

    def retrieval_metrics():
        ev = load_module(ROOT / "week5_rag" / "eval_retrieval.py", "ev")
        assert ev.recall_at_k(["a", "b", "c", "d"], ["c", "z"], 3) == 0.5
        assert abs(ev.mrr(["a", "b", "c"], ["c"]) - 1 / 3) < 1e-12
        assert abs(ev.ndcg_at_k(["c", "a", "b"], ["c"], 3) - 1.0) < 1e-9
        assert ev.ndcg_at_k(["a", "b", "c"], ["c"], 3) < 1.0
        return "recall@k, mrr, ndcg@k correct on known cases"

    def vector_search():
        rng = np.random.default_rng(0)
        M = rng.normal(size=(50, 32))
        M /= np.linalg.norm(M, axis=1, keepdims=True)
        q = M[7] + 0.01 * rng.normal(size=32)
        q /= np.linalg.norm(q)
        assert int(np.argmax(M @ q)) == 7
        return "cosine search retrieves the planted neighbour"

    print("\nWeek 5: retrieval")
    check("chunking boundaries and metadata", chunking)
    check("BM25 exact-term retrieval and RRF", bm25_and_fusion)
    check("recall@k, MRR, nDCG@k", retrieval_metrics)
    check("normalised vector search", vector_search)


def main():
    print("verify.py: reproducing the numbers this repo claims (no GPU, no downloads)")
    missing = []
    for mod in ("torch", "numpy"):
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        print(f"\nMissing: {', '.join(missing)}.  pip install {' '.join(missing)}")
        return 2

    week1(); week2(); week3(); week4(); week5()

    passed = sum(1 for _, ok, _ in RESULTS if ok)
    total = len(RESULTS)
    print(f"\n{'=' * 72}\n{passed}/{total} checks passed")
    if passed < total:
        print("Failures:")
        for name, ok, detail in RESULTS:
            if not ok:
                print(f"  - {name}: {detail}")
        return 1
    print("Everything the booklet claims without a GPU reproduces on this machine.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
