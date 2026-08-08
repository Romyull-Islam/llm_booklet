# week3_finetune/merge.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained(
    "HuggingFaceTB/SmolLM2-135M-Instruct", dtype=torch.bfloat16, device_map="cpu"
)
merged = PeftModel.from_pretrained(base, "models/smollm2-135m-lora/final").merge_and_unload()
merged.save_pretrained("models/smollm2-135m-merged", safe_serialization=True)
AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct").save_pretrained("models/smollm2-135m-merged")
print("merged model written")
