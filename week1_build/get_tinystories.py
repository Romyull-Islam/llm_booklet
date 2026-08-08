"""
get_tinystories.py: build `tinystories.txt` for the scale-up at the end of section 1.7.

The booklet links the TinyStories paper and dataset and then runs
`train_words.py --data tinystories.txt`, but never says how to produce that file.
The dataset ships as parquet on the Hub, not as text, so there is a step missing.
This is it.

    python get_tinystories.py --stories 200000

Why TinyStories works where tiny shakespeare does not: the corpus uses only
vocabulary a young child knows, so a 25M-parameter model has enough capacity to
learn what the words *mean*, not merely how they arrange. Same code, better-matched
data. That is the whole point of the exercise.
"""
import argparse
import os

from datasets import load_dataset

p = argparse.ArgumentParser()
p.add_argument("--stories", type=int, default=200_000,
               help="how many stories to keep. 200k is ~20M words, plenty for 25M params.")
p.add_argument("--out", default="tinystories.txt")
a = p.parse_args()

print(f"streaming roneneldan/TinyStories, taking {a.stories:,} stories ...")
ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)

n = 0
with open(a.out, "w", encoding="utf-8") as f:
    for row in ds:
        story = row["text"].strip()
        if not story:
            continue
        # Blank line between stories: the word tokeniser keeps "\n" as a token, so
        # the model can learn where a story starts and ends.
        f.write(story + "\n\n")
        n += 1
        if n >= a.stories:
            break
        if n % 25_000 == 0:
            print(f"  {n:,} stories")

del ds     # release the streaming reader before we exit

size = os.path.getsize(a.out)
words = sum(len(line.split()) for line in open(a.out, encoding="utf-8"))
print(f"\nwrote {a.out}: {n:,} stories, {words:,} words, {size/1e6:.1f} MB")
print("\nNow train on it (note --n-heads 8: at d_model 512 that gives head_dim 64,")
print("which is what section 2.2's sizing rule asks for):")
print("  python train_words.py --data tinystories.txt --steps 20000 \\")
print("                        --d-model 512 --n-layers 8 --n-heads 8 --min-count 5 \\")
print("                        --seq-len 256 --batch-size 32 --prompt 'Once upon a time'")

# `datasets` streaming leaves a background thread that races the interpreter
# shutdown and prints "Fatal Python error: PyGILState_Release". The file is
# already written and flushed at this point, so exit before that can happen.
os._exit(0)
