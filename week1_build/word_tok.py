"""Word-level tokeniser. The point: a model emitting word tokens CANNOT misspell."""
import re
from collections import Counter

WORD_RE = re.compile(r"[A-Za-z']+|[0-9]+|[^\sA-Za-z0-9']|\n")

class WordTokenizer:
    def __init__(self, text, min_count=2):
        toks = WORD_RE.findall(text)
        counts = Counter(toks)
        self.itos = ["<unk>"] + [t for t, c in counts.most_common() if c >= min_count]
        self.stoi = {t: i for i, t in enumerate(self.itos)}
        self.vocab_size = len(self.itos)
        self.coverage = sum(c for t, c in counts.items() if t in self.stoi) / sum(counts.values())
    def encode(self, s):
        return [self.stoi.get(t, 0) for t in WORD_RE.findall(s)]
    def decode(self, ids):
        out = []
        for i in ids:
            t = self.itos[int(i)]
            if t == "\n": out.append("\n")
            elif re.match(r"[A-Za-z0-9']", t): out.append((" " if out and not out[-1].endswith("\n") else "") + t)
            else: out.append(t)
        return "".join(out).strip()

if __name__ == "__main__":
    text = open("input.txt", encoding="utf-8").read()
    for mc in (1, 2, 3):
        t = WordTokenizer(text, min_count=mc)
        ids = t.encode(text)
        print(f"min_count={mc}: vocab {t.vocab_size:6d} | tokens {len(ids):8,} | coverage {t.coverage:.4f}")
    t = WordTokenizer(text, min_count=2)
    s = "KING RICHARD III:\nThat those bosom.\n"
    assert t.decode(t.encode(s)).replace(" :", ":").replace(" .", ".") .startswith("KING RICHARD III")
    print("\nround-trip:", repr(t.decode(t.encode(s))))
