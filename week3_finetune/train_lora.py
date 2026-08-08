# week3_finetune/train_lora.py
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
    max_length=1024,                     # NOT max_seq_length, that name was removed
    per_device_train_batch_size=8,       # 135M leaves room; raise this before anything else
    gradient_accumulation_steps=2,       # effective batch = 8 * 2 = 16
    gradient_checkpointing=False,        # unnecessary at 135M. Turn on only if you scale up
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
    # packing concatenates several examples into one sequence. It is a real speedup,
    # but TRL warns that only FlashAttention variants keep the samples separated;
    # with sdpa or eager, tokens leak across example boundaries. Off by default here.
    # Turn it on only with attn_implementation="flash_attention_2".
    packing=False,

    # --- observability ---
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=50,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=2,
    seed=42,                             # ablations need a fixed seed
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
