# check_env.py
import sys, platform
import torch

print(f"python        {sys.version.split()[0]}  ({platform.system()})")
print(f"torch         {torch.__version__}")
print(f"cuda built    {torch.version.cuda}")
print(f"cuda available{'':1} {torch.cuda.is_available()}")

if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    total = p.total_memory / 1024**3
    print(f"gpu           {p.name}")
    print(f"vram          {total:.1f} GB")
    print(f"capability    sm_{p.major}{p.minor}")
    print(f"bf16 support  {torch.cuda.is_bf16_supported()}")
    # Prove you can actually allocate and compute.
    a = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    b = a @ a
    torch.cuda.synchronize()
    used = torch.cuda.max_memory_allocated() / 1024**3
    print(f"matmul ok     peak {used:.2f} GB allocated")
else:
    print("NO GPU VISIBLE. Check your driver with `nvidia-smi` before continuing.")

for mod in ["transformers", "trl", "peft", "datasets", "sentence_transformers"]:
    try:
        m = __import__(mod)
        print(f"{mod:<22} {getattr(m, '__version__', 'unknown')}")
    except ImportError:
        print(f"{mod:<22} NOT INSTALLED")
