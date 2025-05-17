# Copyright 2025 Takanori Ishikawa
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import GPTConfig


class RotaryEmbedding(nn.Module):
    """Pre-computes sin / cos tables for RoPE."""

    def __init__(self, head_dim: int, max_pos: int, base: float = 10_000.0):
        super().__init__()

        if head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        t = torch.arange(max_pos, dtype=torch.float)
        freqs = torch.einsum("i,j->ij", t, inv_freq)  # [T, head_dim/2]

        # 交互（even / odd）成分に展開
        emb = torch.cat((freqs, freqs), dim=-1)  # [T, head_dim]
        self.register_buffer("sin", emb.sin(), persistent=False)
        self.register_buffer("cos", emb.cos(), persistent=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x shape = [B, h, T, head_dim]
        T = x.size(-2)
        return (
            self.sin[:T].unsqueeze(0).unsqueeze(0),  # [1,1,T,hd] # type: ignore
            self.cos[:T].unsqueeze(0).unsqueeze(0),  # type: ignore
        )


def apply_rotary(x: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor) -> torch.Tensor:
    # x, sin, cos: [B,h,T,head_dim]
    x_even, x_odd = x[..., ::2], x[..., 1::2]
    rot_even = x_even * cos[..., ::2] - x_odd * sin[..., ::2]
    rot_odd = x_even * sin[..., ::2] + x_odd * cos[..., ::2]
    return torch.stack((rot_even, rot_odd), dim=-1).flatten(-2)


class SelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()

        assert cfg.n_embed % cfg.n_heads == 0

        self.dim = cfg.n_embed
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.n_embed // cfg.n_heads
        self.qkv = nn.Linear(cfg.n_embed, cfg.n_embed * 3, bias=False)
        self.proj = nn.Linear(cfg.n_embed, cfg.n_embed, bias=False)

        # RoPE precomputed
        self.rotary = RotaryEmbedding(self.head_dim, max_pos=cfg.ctx_len)

        # causal mask [T, T] (True means ignore)
        # 上三角行列（対角線含まず）が True になるマスク
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
        qkv = self.qkv(x).contiguous().view(B, T, 3, self.n_heads, self.head_dim)

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
            attn_mask = causal | pad_k  # bool OR
        else:
            attn_mask = causal  # [1, 1, T, T]

        # scaled dot-product attention
        # [B, H, T, D]
        attn_out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
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
