"""
check_labels.py: Appendix B.3's masked-label check, as a runnable file.

B.3 says training loss of exactly 0.0 or nan means every label is masked to -100,
and gives the two-line check:

    batch = next(iter(trainer.get_train_dataloader()))
    print("unmasked label tokens:", (batch["labels"] != -100).sum().item())

but ships no file to run it in, and it is only useful *before* you train. This
builds the same trainer configuration train_lora.py uses and reports the masking
before any optimiser step happens.

Run from the llm-course root:  python week3_finetune/check_labels.py
"""
import torch
from datasets import load_dataset
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"

dataset = load_dataset("json", data_files={
    "train": "data/train.jsonl",
    "validation": "data/val.jsonl",
})

peft_config = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
)

args = SFTConfig(
    output_dir="/tmp/check_labels",
    max_length=1024,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    bf16=True,
    packing=True,
    report_to="none",
    seed=42,
    model_init_kwargs={"dtype": torch.bfloat16},
)

trainer = SFTTrainer(
    model=MODEL, args=args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    peft_config=peft_config,
)

batch = next(iter(trainer.get_train_dataloader()))
labels = batch["labels"]
unmasked = (labels != -100).sum().item()
total = labels.numel()

print()
print(f"  batch shape        {tuple(labels.shape)}")
print(f"  total label tokens {total:,}")
print(f"  unmasked           {unmasked:,}  ({unmasked/total:.1%})")

if unmasked == 0:
    print("\n  FAIL: every label is masked. Loss will be 0.0 or nan and means nothing.")
    print("  Fix the dataset format, or drop assistant_only_loss. See appendix B.3.")
    raise SystemExit(1)

print("\n  PASS: the model has real tokens to learn from.")

# The other half of B.3 that the booklet does not mention: masking can be present
# but wrong. Decode one example so you can see what loss is actually computed on.
tok = trainer.processing_class
row = labels[0]
kept = row[row != -100]
print(f"\n  first row, tokens loss is computed on ({len(kept)} of {len(row)}):")
print("   ", repr(tok.decode(kept[:60], skip_special_tokens=False))[:300])
