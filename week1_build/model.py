"""
model.py: a decoder-only transformer built from primitives.

Everything a modern LLM has, and nothing it does not:
  RMSNorm, rotary position embeddings, grouped-query attention with a KV cache,
  SwiGLU feed-forward, pre-norm residual blocks, weight tying.
Set n_experts > 1 and the feed-forward becomes a mixture-of-experts layer.

No transformers, no accelerate. Only torch.
"""
from dataclasses import dataclass
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    vocab_size: int = 256
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 8
    n_kv_heads: int = 2        # < n_heads gives grouped-query attention
    d_ff: int = 704            # ~2.75 * d_model, the usual SwiGLU ratio
    max_seq_len: int = 256
    rope_theta: float = 10000.0
    dropout: float = 0.0
    # Mixture of experts. n_experts = 1 means a plain dense MLP.
    n_experts: int = 1
    n_experts_active: int = 2
    aux_loss_coef: float = 0.01


# ---------------------------------------------------------------- normalisation

class RMSNorm(nn.Module):
    """LayerNorm without the mean subtraction. Cheaper, and works just as well."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() * rms).type_as(x) * self.weight


# ------------------------------------------------------------------- positions

def build_rope_cache(seq_len, head_dim, theta, device, dtype):
    """Precompute cos and sin tables for rotary position embeddings."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)               # (seq_len, head_dim/2)
    return freqs.cos().to(dtype), freqs.sin().to(dtype)


def apply_rope(x, cos, sin):
    """x: (batch, heads, seq, head_dim). Rotates each pair of dimensions by m*theta."""
    x1, x2 = x.chunk(2, dim=-1)
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


# ------------------------------------------------------------------- attention

class Attention(nn.Module):
    """Grouped-query attention with an optional KV cache for incremental decoding."""

    def __init__(self, cfg: Config):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        assert cfg.n_heads % cfg.n_kv_heads == 0
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.n_rep = cfg.n_heads // cfg.n_kv_heads
        self.head_dim = cfg.d_model // cfg.n_heads

        self.q_proj = nn.Linear(cfg.d_model, cfg.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.n_heads * self.head_dim, cfg.d_model, bias=False)

        # Normally `is_causal` is decided per call from T. Under torch.export with a
        # dynamic sequence axis T is symbolic, so `T > 1` is a SymBool and
        # scaled_dot_product_attention rejects it. Set this to a real bool to pin
        # the choice for export. None keeps the runtime behaviour.
        self.force_causal = None

    def forward(self, x, cos, sin, cache=None):
        B, T, _ = x.shape

        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # The KV cache: append this step's K and V to everything computed before.
        # This is what turns O(T^2) regeneration into O(T) incremental decoding.
        if cache is not None:
            past_k, past_v = cache
            if past_k is not None:
                k = torch.cat([past_k, k], dim=2)
                v = torch.cat([past_v, v], dim=2)
            new_cache = (k, v)
        else:
            new_cache = None

        # Grouped-query attention: each KV head is shared by n_rep query heads.
        # This is why the cache shrinks by n_rep compared with multi-head attention.
        k = k.repeat_interleave(self.n_rep, dim=1)
        v = v.repeat_interleave(self.n_rep, dim=1)

        # Causal only when processing more than one position. During decoding
        # the single new query legitimately attends to the whole cache.
        causal = (T > 1) if self.force_causal is None else self.force_causal
        out = F.scaled_dot_product_attention(q, k, v, is_causal=causal)

        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.o_proj(out), new_cache


# ----------------------------------------------------------------- feedforward

class SwiGLU(nn.Module):
    """gate(x) * up(x) then down. Three matrices, not two, which is why d_ff ~ 2.75x."""

    def __init__(self, d_model, d_ff):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class MoE(nn.Module):
    """Sparse mixture of experts: a router picks top-k experts per token.

    Total parameters grow with n_experts. Compute grows only with n_experts_active.
    That decoupling is the entire point of the architecture.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.n_experts = cfg.n_experts
        self.top_k = cfg.n_experts_active
        self.router = nn.Linear(cfg.d_model, cfg.n_experts, bias=False)
        self.experts = nn.ModuleList([SwiGLU(cfg.d_model, cfg.d_ff) for _ in range(cfg.n_experts)])
        self.aux_loss = torch.tensor(0.0)

    def forward(self, x):
        B, T, D = x.shape
        flat = x.view(-1, D)                                  # (N, D)

        logits = self.router(flat)                            # (N, E)
        probs = F.softmax(logits, dim=-1)
        topk_probs, topk_idx = probs.topk(self.top_k, dim=-1)  # (N, k)
        topk_probs = topk_probs / topk_probs.sum(-1, keepdim=True)

        out = torch.zeros_like(flat)
        for e in range(self.n_experts):
            mask = (topk_idx == e)                            # (N, k)
            if not mask.any():
                continue
            token_idx, slot = mask.nonzero(as_tuple=True)
            weight = topk_probs[token_idx, slot].unsqueeze(-1)
            out.index_add_(0, token_idx, self.experts[e](flat[token_idx]) * weight)

        # Load-balancing auxiliary loss (Switch Transformer form).
        # Without it the router collapses onto a couple of favourite experts.
        frac_tokens = torch.zeros(self.n_experts, device=x.device, dtype=probs.dtype)
        frac_tokens.index_add_(0, topk_idx.reshape(-1),
                               torch.ones(topk_idx.numel(), device=x.device, dtype=probs.dtype))
        frac_tokens = frac_tokens / topk_idx.numel()
        frac_prob = probs.mean(dim=0)
        self.aux_loss = self.n_experts * (frac_tokens * frac_prob).sum()

        return out.view(B, T, D)


# ----------------------------------------------------------------------- block

class Block(nn.Module):
    """Pre-norm residual block. Norm before the sublayer, not after: this is what
    made deep transformers trainable without a warmup crutch."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.ffn_norm = RMSNorm(cfg.d_model)
        self.ffn = MoE(cfg) if cfg.n_experts > 1 else SwiGLU(cfg.d_model, cfg.d_ff)

    def forward(self, x, cos, sin, cache=None):
        h, new_cache = self.attn(self.attn_norm(x), cos, sin, cache)
        x = x + h
        x = x + self.ffn(self.ffn_norm(x))
        return x, new_cache


class Transformer(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight        # weight tying
        self._rope = None
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def _rope_cache(self, device, dtype, needed):
        """Grow the RoPE table on demand.

        Generating past `max_seq_len` is allowed here so the model does not crash,
        but the model was never trained at those positions, so quality degrades.
        Making that degradation graceful is the length-extrapolation problem.
        """
        if (self._rope is None or self._rope[0].device != device
                or self._rope[0].shape[0] < needed):
            size = max(needed, self.cfg.max_seq_len)
            self._rope = build_rope_cache(size, self.cfg.d_model // self.cfg.n_heads,
                                          self.cfg.rope_theta, device, dtype)
        return self._rope

    def forward(self, idx, targets=None, caches=None, pos_offset=0):
        B, T = idx.shape
        x = self.embed(idx)
        cos_all, sin_all = self._rope_cache(x.device, x.dtype, pos_offset + T)
        cos = cos_all[pos_offset:pos_offset + T]
        sin = sin_all[pos_offset:pos_offset + T]

        new_caches = []
        for i, block in enumerate(self.blocks):
            cache = caches[i] if caches is not None else None
            x, nc = block(x, cos, sin, cache)
            new_caches.append(nc)

        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1))
            if self.cfg.n_experts > 1:
                aux = sum(b.ffn.aux_loss for b in self.blocks)
                loss = loss + self.cfg.aux_loss_coef * aux
        return logits, loss, new_caches

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, use_cache=True,
                 ban_tokens=None):
        """ban_tokens: token ids the sampler must never emit. The word-level
        tokeniser needs this for <unk> (index 0), which would otherwise show up
        in the output as literal '<unk>' text."""
        self.eval()
        if not use_cache:
            # The naive path: re-run the whole prefix every single step. It must NOT
            # slice to max_seq_len -- capping it makes this path linear too, so the
            # cache appears to lose on measurement, and past max_seq_len the two
            # paths stop producing the same tokens. RoPE grows on demand, so the
            # full prefix is safe at any length.
            for _ in range(max_new_tokens):
                logits, _, _ = self.forward(idx)
                idx = torch.cat([idx, self._sample(logits[:, -1], temperature, top_k, ban_tokens)], dim=1)
            return idx

        # The cached path: prefill once, then feed one token at a time.
        caches = [(None, None)] * self.cfg.n_layers
        logits, _, caches = self.forward(idx, caches=caches, pos_offset=0)
        pos = idx.shape[1]
        for _ in range(max_new_tokens):
            nxt = self._sample(logits[:, -1], temperature, top_k, ban_tokens)
            idx = torch.cat([idx, nxt], dim=1)
            logits, _, caches = self.forward(nxt, caches=caches, pos_offset=pos)
            pos += 1
        return idx

    @staticmethod
    def _sample(logits, temperature, top_k, ban_tokens=None):
        if ban_tokens is not None:
            logits = logits.clone()
            logits[:, ban_tokens] = float("-inf")
        if temperature == 0.0:
            return logits.argmax(-1, keepdim=True)
        logits = logits / temperature
        if top_k is not None:
            v, _ = logits.topk(min(top_k, logits.size(-1)), dim=-1)
            logits = logits.masked_fill(logits < v[:, [-1]], float("-inf"))
        return torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)

    def param_report(self):
        total = sum(p.numel() for p in self.parameters())
        expert = sum(p.numel() for n, p in self.named_parameters() if ".experts." in n)
        if self.cfg.n_experts > 1:
            active = total - expert + expert * self.cfg.n_experts_active / self.cfg.n_experts
        else:
            active = total
        return {"total": total, "active_per_token": int(active),
                "embedding": self.embed.weight.numel()}
