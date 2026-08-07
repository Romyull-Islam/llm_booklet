# week3_finetune/full_vs_lora.py
import json, time, torch
from datasets import load_dataset
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
dataset = load_dataset("json", data_files={"train": "data/train.jsonl",
                                           "validation": "data/val.jsonl"})

def run(tag, peft_config, lr):
    args = SFTConfig(
        output_dir=f"models/smollm2-135m-{tag}",
        max_length=1024,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        num_train_epochs=1,
        bf16=True,
        packing=True,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        report_to="none",
        seed=42,
        model_init_kwargs={"dtype": torch.bfloat16},
    )
    trainer = SFTTrainer(model=MODEL, args=args,
                         train_dataset=dataset["train"],
                         eval_dataset=dataset["validation"],
                         peft_config=peft_config)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    trainer.train()
    stats = {
        "tag": tag,
        "trainable": sum(p.numel() for p in trainer.model.parameters() if p.requires_grad),
        "eval_loss": trainer.evaluate()["eval_loss"],
        "seconds": round(time.time() - t0, 1),
        "peak_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 2),
    }
    trainer.save_model(f"models/smollm2-135m-{tag}/final")
    del trainer
    torch.cuda.empty_cache()
    return stats

results = [
    run("lora-r8",  LoraConfig(r=8,  lora_alpha=16, task_type="CAUSAL_LM",
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]), 1e-4),
    run("lora-r32", LoraConfig(r=32, lora_alpha=64, task_type="CAUSAL_LM",
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]), 1e-4),
    run("full",     None, 1e-5),          # full FT wants ~10x lower LR than LoRA
]

print(f"{'config':<10}{'trainable':>12}{'eval_loss':>12}{'seconds':>10}{'peak GB':>10}")
for r in results:
    print(f"{r['tag']:<10}{r['trainable']:>12,}{r['eval_loss']:>12.4f}{r['seconds']:>10.1f}{r['peak_gb']:>10.2f}")
json.dump(results, open("results_full_vs_lora.json", "w"), indent=2)
