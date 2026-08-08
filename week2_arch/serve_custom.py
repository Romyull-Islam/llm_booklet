# week2_arch/serve_custom.py
import os
import sys
import torch
from fastapi import FastAPI
from pydantic import BaseModel

# This file lives in week2_arch/ but the model and tokeniser it serves are the
# ones you wrote in week1_build/, and the checkpoint is written there too.
WEEK1 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "week1_build")
sys.path.insert(0, WEEK1)

from model import Config, Transformer          # noqa: E402
from train import CharTokenizer                # noqa: E402

app = FastAPI(title="Custom LM")

CKPT = os.environ.get("CKPT", os.path.join(WEEK1, "ckpt_dense.pt"))
ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
cfg = ckpt["cfg"]
model = Transformer(cfg)
model.load_state_dict(ckpt["model"])
model.eval()

tok = CharTokenizer("")
tok.chars = ckpt["chars"]
tok.stoi = {c: i for i, c in enumerate(tok.chars)}
tok.itos = {i: c for c, i in tok.stoi.items()}
tok.vocab_size = len(tok.chars)

class Req(BaseModel):
    prompt: str
    max_new_tokens: int = 200
    temperature: float = 0.8
    top_k: int = 40

@app.get("/health")
def health():
    return {"status": "ok", "params": model.param_report()}

@app.post("/generate")
def generate(r: Req):
    ids = torch.tensor([tok.encode(r.prompt)])
    out = model.generate(ids, r.max_new_tokens, temperature=r.temperature, top_k=r.top_k)
    return {"completion": tok.decode(out[0].tolist())}
