from dataclasses import dataclass
from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rope import RotaryEmbedding, apply_rotary


@dataclass(frozen=True)
class GPTConfig:
    vocab_size: int  # BPE vocab (same as GPT‑2)
    ctx_len: int  # maximum sequence length
    n_embed: int  # model dimension (d_model)
    n_heads: int  # number of attention heads
    n_layers: int  # transformer blocks
    mlp_ratio: int  # feed‑forward expansion ratio
    dropout: float = 0.0


class SelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()

        assert cfg.n_embed % cfg.n_heads == 0

        self.dim = cfg.n_embed
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.n_embed // cfg.n_heads
        self.qkv = nn.Linear(cfg.n_embed, cfg.n_embed * 3, bias=False)
        self.proj = nn.Linear(cfg.n_embed, cfg.n_embed, bias=False)
        self.dropout = cfg.dropout

        # RoPE precomputed
        self.rotary = RotaryEmbedding(self.head_dim, max_pos=cfg.ctx_len)

        # causal mask [T, T]
        #
        # NOTE: MPS は SDPA に渡す attn_mask を bool か "Q・K と同じ dtype" しか受け付けな
        #       い。 float の -inf や -1e4 を使うと、暗黙のキャストで NaN になってしまう問
        #       題もあるため、 bool の mask を用意
        causal = torch.triu(torch.ones(cfg.ctx_len, cfg.ctx_len, dtype=torch.bool), 1)

        self.register_buffer(
            "causal_mask",
            causal,
            persistent=False,
        )

    def forward(
        self, x: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        # x: [B, T, D]
        B, T, _ = x.shape
        # project q, k, v
        qkv = self.qkv(x).view(B, T, 3, self.n_heads, self.head_dim)

        q, k, v = qkv.unbind(dim=2)  # each [B, T, H, D]
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))  # [B, H, T, D]

        # apply RoPE
        sin, cos = self.rotary(q)
        q = apply_rotary(q, sin, cos)
        k = apply_rotary(k, sin, cos)

        # build combined attention mask
        # causal: [T, T] -> [1, 1, T, T]
        causal = cast(torch.Tensor, self.causal_mask)[:T, :T].unsqueeze(0).unsqueeze(0)

        if attention_mask is not None:
            # attention_mask: 1=token, 0=pad  → pad 位置に -1e4
            # -inf を bfloat16 / fp16 に直接変換すると NaN が出る。特に MPS や一部 GPU のハードウェア実装で顕著
            # pad = (attention_mask == 0).view(B, 1, 1, T)
            # attn_mask = causal | pad  # bool OR

            pad_k = (attention_mask == 0).to(torch.bool).view(B, 1, 1, T)  # key 方向
            pad_q = pad_k.transpose(-1, -2)  # query 方向
            attn_mask = causal | pad_k | pad_q  # bool OR
        else:
            attn_mask = causal  # [1, 1, T, T]

        # scaled dot-product attention
        # [B, H, T, D]
        attn_out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout,
            is_causal=False,
        )

        if not torch.isfinite(attn_out).all():
            raise ValueError("NaN after scaled_dot_product_attention", attn_out)

        # reshape and project
        out = attn_out.transpose(1, 2).contiguous().view(B, T, self.dim)
        out = self.proj(out)

        return out


class FeedForward(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        inner = cfg.n_embed * cfg.mlp_ratio
        self.fc1 = nn.Linear(cfg.n_embed, inner)
        self.fc2 = nn.Linear(inner, cfg.n_embed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class TransformerBlock(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embed)
        self.attn = SelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embed)
        self.mlp = FeedForward(cfg)

    def forward(
        self, x: torch.Tensor, *, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        # pass attention_mask to SelfAttention
        x = self.ln1(x)
        x = x + self.attn(x, attention_mask=attention_mask)
        if not torch.isfinite(x).all():
            raise ValueError("NaN after attn", x)

        x = self.ln2(x)
        x = x + self.mlp(x)

        return x


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

    def forward(
        self,
        input_ids: torch.LongTensor,
        *,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # input_ids: [B, T], attention_mask: [B, T]
        x = self.tok_emb(input_ids)  # [B, T, D]
        for block in self.blocks:
            x = block(x, attention_mask=attention_mask)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits

    @torch.no_grad()
    def _generate(self, idx: torch.Tensor, max_new_tokens: int = 32):
        """Greedy text generation (for demo)."""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.ctx_len :]
            logits = self(idx_cond)[:, -1, :]  # [B,vocab]
            next_tok = logits.argmax(dim=-1, keepdim=True)
            idx = torch.cat([idx, next_tok], dim=1)
        return idx


# PyTorch のデフォルト初期化だと、|logit| >= 10 となってしまい、スケール爆発 -> CE loss が高止まりしてしまう
# 0.02 trunc-normal で初期化
def init_weights(module: nn.Module) -> None:
    if isinstance(module, (nn.Linear, nn.Embedding)):
        nn.init.trunc_normal_(module.weight, mean=0.0, std=0.02, a=-0.04, b=0.04)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)
