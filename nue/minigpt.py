from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rope import RotaryEmbedding, apply_rotary


@dataclass(frozen=True)
class GPTConfig:
    vocab_size: int  # BPE vocab (same as GPT‑2)
    ctx_len: int  # maximum sequence length
    n_embed: int  # model dimension (d_model)
    n_heads: int  # number of attention heads (keep =1 for tiny)
    n_layers: int  # transformer blocks (1 → the minimal GPT!)
    mlp_ratio: int  # feed‑forward expansion ratio (2 instead of 4)
    dropout: float = 0.0  # no dropout ⇒ deterministic inference


class SelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.dim = cfg.n_embed
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.n_embed // cfg.n_heads
        self.qkv = nn.Linear(cfg.n_embed, cfg.n_embed * 3, bias=False)
        self.proj = nn.Linear(cfg.n_embed, cfg.n_embed, bias=False)
        self.dropout = cfg.dropout

        # RoPE テーブルを事前計算
        self.rotary = RotaryEmbedding(self.head_dim, max_pos=cfg.ctx_len)

        # causal-mask は is_causal=True で済む
        self.register_buffer(
            "casual_mask",
            torch.triu(
                torch.full((cfg.ctx_len, cfg.ctx_len), float("-inf")), diagonal=1
            ),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # [B,T,h,hd]
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))  # [B,h,T,hd]

        # --- RoPE を適用 ---
        sin, cos = self.rotary(q)
        q = apply_rotary(q, sin, cos)
        k = apply_rotary(k, sin, cos)

        # --- Attention ---
        attn_out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=self.casual_mask[:T, :T],  # [T,T] or None # type: ignore
            dropout_p=self.dropout,
            is_causal=False,  # mask を渡すので False
        )  # [B,h,T,hd]

        out = attn_out.transpose(1, 2).contiguous().view(B, T, self.dim)
        return self.proj(out)


class FeedForward(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        inner = cfg.n_embed * cfg.mlp_ratio
        self.fc1 = nn.Linear(cfg.n_embed, inner)
        self.fc2 = nn.Linear(inner, cfg.n_embed)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class TransformerBlock(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embed)
        self.attn = SelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embed)
        self.mlp = FeedForward(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


# ------------------------------------------------------------
#  Minimal GPT model
# ------------------------------------------------------------
class MinimalGPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embed)
        self.blocks = nn.ModuleList(
            [TransformerBlock(cfg) for _ in range(cfg.n_layers)]
        )
        self.ln_f = nn.LayerNorm(cfg.n_embed)
        self.head = nn.Linear(cfg.n_embed, cfg.vocab_size, bias=False)

        # weight tying
        self.head.weight = self.tok_emb.weight

    @torch.no_grad()
    def _generate(self, idx: torch.Tensor, max_new_tokens: int = 32):
        """Greedy text generation (for demo)."""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.ctx_len :]
            logits = self(idx_cond)[:, -1, :]  # [B,vocab]
            next_tok = logits.argmax(dim=-1, keepdim=True)
            idx = torch.cat([idx, next_tok], dim=1)
        return idx

    def forward(self, idx: torch.LongTensor):  # idx:[B,T]
        B, T = idx.shape
        tok = self.tok_emb(idx)  # [B,T,d]
        x = tok
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.head(x)


# PyTorch のデフォルト初期化だと、|logit| >= 10 となってしまい、スケール爆発 -> CE loss が高止まりしてしまう
# 0.02 trunc-normal で初期化
def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        nn.init.trunc_normal_(module.weight, mean=0.0, std=0.02, a=-0.04, b=0.04)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)
