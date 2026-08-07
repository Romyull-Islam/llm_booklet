# week2_arch/serve_custom.py
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from model import Config, Transformer
from train import CharTokenizer

app = FastAPI(title="Custom LM")

ckpt = torch.load("ckpt_dense.pt", map_location="cpu", weights_only=False)
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
