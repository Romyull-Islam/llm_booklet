# week3_finetune/train_vlm_lora.py
from datasets import load_dataset
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

trainer = SFTTrainer(
    model="HuggingFaceTB/SmolVLM-256M-Instruct",
    args=SFTConfig(
        output_dir="models/smolvlm-256m-lora",
        max_length=None,               # critical: truncation would cut image tokens
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        learning_rate=1e-4,
        num_train_epochs=1,
        bf16=True,
        logging_steps=10,
        report_to="none",
    ),
    train_dataset=load_dataset("trl-lib/llava-instruct-mix", split="train[:500]"),
    # target_modules is NOT optional here. PEFT auto-infers it from a table of ~38
    # model types; Idefics3 (SmolVLM) is not in that table, so omitting this raises
    # "Please specify `target_modules` or `target_parameters` in `peft_config`".
    # These names hit the Llama-style text tower, which is where the adaptation matters.
    peft_config=LoraConfig(
        r=16, lora_alpha=32, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    ),
)
trainer.train()
