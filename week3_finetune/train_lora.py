# week2_finetune/train_lora.py
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
    r=16,
    lora_alpha=32,                 # convention: 2 * r
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
)

args = SFTConfig(
    output_dir="models/smollm2-135m-lora",

    # --- memory. tune these first when you hit OOM ---
    max_length=2048,                     # NOT max_seq_length, that name was removed
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,       # effective batch = 2 * 8 = 16
    gradient_checkpointing=True,         # ~40% less activation memory, ~20% slower
    bf16=True,
    model_init_kwargs={"dtype": torch.bfloat16},

    # --- optimisation ---
    learning_rate=1e-4,                  # LoRA wants ~10x a full fine-tune LR
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    num_train_epochs=1,
    max_grad_norm=1.0,
    optim="adamw_torch_fused",

    # --- efficiency ---
    packing=True,                        # pack short examples together, big speedup

    # --- observability ---
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=200,
    save_total_limit=2,
    report_to="none",                    # set to "trackio" or "wandb" for curves
)

trainer = SFTTrainer(
    model=MODEL,
    args=args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    peft_config=peft_config,
)

trainer.train()
trainer.save_model("models/smollm2-135m-lora/final")
print("saved adapter to models/smollm2-135m-lora/final")
