# week1_serve/04_vlm.py
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor

MODEL = "HuggingFaceTB/SmolVLM-256M-Instruct"   # requires transformers >= 4.57

model = AutoModelForVision2Seq.from_pretrained(
    MODEL, dtype=torch.bfloat16, device_map="auto"
)
processor = AutoProcessor.from_pretrained(MODEL)

messages = [{
    "role": "user",
    "content": [
        {"type": "image", "image": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"},
        {"type": "text", "text": "Describe this image in two sentences, then list every object you can identify."},
    ],
}]

inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)

with torch.inference_mode():
    out = model.generate(**inputs, max_new_tokens=256)

trimmed = out[0][inputs["input_ids"].shape[-1]:]
print(processor.decode(trimmed, skip_special_tokens=True))
print(f"peak VRAM {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")
