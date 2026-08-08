"""
export_onnx.py: section 2.4 option B, export the Week 1 model to ONNX.

The booklet gives the torch.onnx.export call but not a runnable file, and the
call as printed exports Transformer.forward directly. That signature returns
(logits, loss, new_caches) where loss is None and new_caches is a list of Nones,
which the exporter cannot represent. Pass --raw to see that failure, or run
without it to export through a thin logits-only wrapper.
"""
import os
import sys
import argparse
import torch
import torch.nn as nn

WEEK1 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "week1_build")
sys.path.insert(0, WEEK1)
from model import Transformer                  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default=os.path.join(WEEK1, "ckpt_dense.pt"))
ap.add_argument("--out", default="model.onnx")
ap.add_argument("--raw", action="store_true", help="export forward() as the booklet prints it")
args = ap.parse_args()

ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
cfg = ck["cfg"]
model = Transformer(cfg)
model.load_state_dict(ck["model"])
model.eval()


class LogitsOnly(nn.Module):
    """ONNX wants a tensor (or tuple of tensors) out. Drop loss and the caches."""

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, input_ids):
        logits, _, _ = self.m(input_ids)
        return logits


export_target = model if args.raw else LogitsOnly(model)

if not args.raw:
    # Pin the causal choice. The exported graph is the prefill path, which is
    # always causal; leaving it as `T > 1` makes it a SymBool on the dynamic axis.
    for block in model.blocks:
        block.attn.force_causal = True

dummy = torch.randint(0, cfg.vocab_size, (1, 32))

torch.onnx.export(export_target, (dummy,), args.out,
                  input_names=["input_ids"], output_names=["logits"],
                  dynamic_axes={"input_ids": {0: "batch", 1: "seq"},
                                "logits": {0: "batch", 1: "seq"}},
                  opset_version=17)

size = os.path.getsize(args.out) / 1024**2
print(f"exported {args.out}  ({size:.2f} MB)")

# Prove the exported graph agrees with PyTorch.
try:
    import onnxruntime as ort
    import numpy as np
except ImportError:
    print("onnxruntime not installed; skipping numerical check")
    sys.exit(0)

sess = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
for seq_len in (32, 64):                      # exercise the dynamic axis
    x = torch.randint(0, cfg.vocab_size, (1, seq_len))
    with torch.no_grad():
        ref, _, _ = model(x)
    got = sess.run(["logits"], {"input_ids": x.numpy()})[0]
    diff = np.abs(ref.numpy() - got).max()
    print(f"seq_len {seq_len:>3}: max |onnx - torch| = {diff:.2e}"
          f"  {'PASS' if diff < 1e-4 else 'FAIL'}")
