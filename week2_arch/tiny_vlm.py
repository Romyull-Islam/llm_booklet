# week2_arch/tiny_vlm.py
"""Toy VLM: prove the vision-token splicing works end to end, no pretrained weights."""
import torch, torch.nn as nn
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent / "week1_build"))
from model import Config, Transformer

class PatchEncoder(nn.Module):
    """Stand-in for a real ViT: cut the image into patches, embed each one."""
    def __init__(self, img_size=32, patch=8, d_vision=64):
        super().__init__()
        self.proj = nn.Conv2d(3, d_vision, kernel_size=patch, stride=patch)
        self.n_patches = (img_size // patch) ** 2
    def forward(self, img):                       # (B, 3, H, W)
        x = self.proj(img)                        # (B, d_vision, H/p, W/p)
        return x.flatten(2).transpose(1, 2)       # (B, n_patches, d_vision)

class Projector(nn.Module):
    """The whole trick: map vision features into the LM's token space."""
    def __init__(self, d_vision, d_model):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_vision, d_model), nn.GELU(),
                                 nn.Linear(d_model, d_model))
    def forward(self, feats): return self.net(feats)

class TinyVLM(nn.Module):
    def __init__(self, cfg, d_vision=64):
        super().__init__()
        self.lm = Transformer(cfg)
        self.vision = PatchEncoder(d_vision=d_vision)
        self.projector = Projector(d_vision, cfg.d_model)
    def forward(self, img, text_ids, targets=None):
        vis = self.projector(self.vision(img))            # (B, P, d_model)
        txt = self.lm.embed(text_ids)                     # (B, T, d_model)
        x = torch.cat([vis, txt], dim=1)                  # image tokens then text
        # run the LM stack directly on the spliced embeddings
        cos, sin = self.lm._rope_cache(x.device, x.dtype, x.shape[1])
        cos, sin = cos[:x.shape[1]], sin[:x.shape[1]]
        for b in self.lm.blocks:
            x, _ = b(x, cos, sin, None)
        logits = self.lm.lm_head(self.lm.norm(x))
        loss = None
        if targets is not None:
            text_logits = logits[:, vis.shape[1]:, :]     # only score text positions
            loss = nn.functional.cross_entropy(
                text_logits[:, :-1].reshape(-1, logits.size(-1)), targets[:, 1:].reshape(-1))
        return logits, loss


if __name__ == "__main__":
    torch.manual_seed(0)
    cfg = Config(vocab_size=65, d_model=128, n_layers=2, n_heads=4, n_kv_heads=2, d_ff=352, max_seq_len=64)
    m = TinyVLM(cfg)
    img = torch.randn(2, 3, 32, 32)
    ids = torch.randint(0, 65, (2, 12))
    logits, loss = m(img, ids, targets=ids)
    print("patches per image:", m.vision.n_patches)
    print("logits", tuple(logits.shape), "= 16 image tokens + 12 text tokens")
    assert logits.shape == (2, 16 + 12, 65)
    print("loss", round(loss.item(), 4), "vs ln(65) =", round(torch.log(torch.tensor(65.)).item(), 4))

    # gradient must reach the vision encoder through the projector
    loss.backward()
    g = m.vision.proj.weight.grad
    print("vision encoder grad norm:", round(g.norm().item(), 6))
    assert g is not None and g.norm().item() > 0
    print("TINY VLM: gradient flows image -> projector -> LM -> loss. PASS")

    n_lm = sum(p.numel() for p in m.lm.parameters())
    n_v  = sum(p.numel() for p in m.vision.parameters())
    n_p  = sum(p.numel() for p in m.projector.parameters())
    print(f"params: lm {n_lm:,}  vision {n_v:,}  projector {n_p:,}")
