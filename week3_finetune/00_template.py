# week3_finetune/00_template.py
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")

messages = [
    {"role": "system", "content": "You are a precise technical assistant."},
    {"role": "user", "content": "Explain what a KV cache is."},
]

print("=== EXACT STRING FED TO THE MODEL ===")
print(repr(tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)))
print("=== WITHOUT the generation prompt ===")
print(repr(tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)))
print("eos:", tok.eos_token, tok.eos_token_id)
