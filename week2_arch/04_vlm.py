# week2_arch/04_vlm.py
import torch
# transformers v5 removed AutoModelForVision2Seq; it is AutoModelForImageTextToText
# now. Week 0 installs "transformers>=4.57", which resolves to v5, so the booklet's
# import fails on a fresh environment. Fall back so both versions work.
try:
    from transformers import AutoModelForImageTextToText as AutoVLM
except ImportError:                                  # transformers 4.x
    from transformers import AutoModelForVision2Seq as AutoVLM
from transformers import AutoProcessor

MODEL = "HuggingFaceTB/SmolVLM-256M-Instruct"   # 256M, runs in under 1 GB of VRAM

model = AutoVLM.from_pretrained(
    MODEL, dtype=torch.bfloat16, device_map="auto"
)
processor = AutoProcessor.from_pretrained(MODEL)

IMAGE = ("https://huggingface.co/datasets/huggingface/documentation-images/"
         "resolve/main/pipeline-cat-chonk.jpeg")


def ask(prompt, longest_edge=None):
    if longest_edge is not None:
        # SmolVLM controls image tokens through the processor's resize target.
        processor.image_processor.size = {"longest_edge": longest_edge}
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": IMAGE},
            {"type": "text", "text": prompt},
        ],
    }]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=256)
    trimmed = out[0][inputs["input_ids"].shape[-1]:]
    return (processor.decode(trimmed, skip_special_tokens=True),
            inputs["input_ids"].shape[-1],
            torch.cuda.max_memory_allocated() / 1024**3)


# Keep the ask small. A 256M VLM writes a good short caption and falls apart
# when you ask it to enumerate every object. Match the prompt to the model.
text, n_tok, vram = ask("Describe this image in one sentence.")
print("=== the ask the booklet recommends ===")
print(text)
print(f"prompt tokens {n_tok} | peak VRAM {vram:.2f} GB")

# The prompt the booklet warns against, to check that the warning is real.
text, n_tok, vram = ask("Describe this image in two sentences, then list every "
                        "object you can identify.")
print("\n=== the ask the booklet warns against ===")
print(text)
print(f"prompt tokens {n_tok} | peak VRAM {vram:.2f} GB")

# The VRAM knob: lower this first when a VLM runs out of memory.
print("\n=== image-token / VRAM knob (processor resize target) ===")
for edge in (2048, 1024, 512, 256):
    text, n_tok, vram = ask("Describe this image in one sentence.", longest_edge=edge)
    print(f"longest_edge {edge:>5} | prompt tokens {n_tok:>5} | peak VRAM {vram:.2f} GB "
          f"| {text.strip()[:70]}")
