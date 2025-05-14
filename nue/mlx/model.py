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
import math
from typing import cast

import mlx.core as mx
import mlx.nn as nn

from nue.model.base import GPTConfig


class FeedForward(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        inner = cfg.n_embed * cfg.mlp_ratio
        self.fc1 = nn.Linear(cfg.n_embed, inner)
        self.fc2 = nn.Linear(inner, cfg.n_embed)

    def __call__(self, x: mx.array) -> mx.array:
        return self.fc2(nn.gelu(self.fc1(x)))


class SelfAttention(nn.Module):
    """
    Decoder-style self-attention with RoPE (rotary positional embedding)
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embed % config.n_heads == 0, (
            "n_embed must be divisible by n_heads"
        )
        self.n_heads = config.n_heads
        self.head_dim = config.n_embed // config.n_heads

        # QKV projection
        self.q_proj = nn.Linear(config.n_embed, config.n_embed, bias=False)
        self.k_proj = nn.Linear(config.n_embed, config.n_embed, bias=False)
        self.v_proj = nn.Linear(config.n_embed, config.n_embed, bias=False)

        # Output projection
        self.out_proj = nn.Linear(config.n_embed, config.n_embed, bias=False)

        # RoPE
        self.rope = nn.RoPE(self.head_dim)

        # Causal mask template (float32)
        self.causal = nn.MultiHeadAttention.create_additive_causal_mask(config.ctx_len)

    def __call__(self, x: mx.array, attention_mask: mx.array | None = None):
        # x: [B, L, D]
        # mask: **boolean mask** [B|1, H|1, L_q, L_k]
        B, L, _ = x.shape
        H, Hd = self.n_heads, self.head_dim

        # 1) QKV projection
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # 2) (B, L, H, Hd) → (B, H, L, Hd)
        q = q.reshape(B, L, H, Hd).transpose(0, 2, 1, 3)
        k = k.reshape(B, L, H, Hd).transpose(0, 2, 1, 3)
        v = v.reshape(B, L, H, Hd).transpose(0, 2, 1, 3)

        # 3) RoPE
        q = self.rope(q)
        k = self.rope(k)

        # 4) Prepare attention mask
        if attention_mask is not None:
            # boolean mask -> additive mask to reduce conversion cost in SDPA kernel
            pad = cast(mx.array, attention_mask == 0)  # [B, L]、True=pad
            pad = mx.expand_dims(pad, (1, 2)).astype(q.dtype) * -1e9  # [B,1,1,L]

            causal = self.causal[:L, :L][None, None, ...]  # [1,1,L,L]
            attention_mask = causal + pad
        else:
            attention_mask = self.causal[:L, :L][None, None, ...].astype(
                q.dtype
            )  # [1,1,L,L]

        # 5) scaled dot product attention
        out = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=math.sqrt(1.0 / Hd), mask=attention_mask
        )

        out = out.transpose(0, 2, 1, 3).reshape(B, L, H * Hd)  # (B,L,D)
        out = self.out_proj(out)

        return out


class TransformerBlock(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embed)
        self.attn = SelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embed)
        self.mlp = FeedForward(cfg)

    def __call__(
        self, x: mx.array, *, attention_mask: mx.array | None = None
    ) -> mx.array:
        # pass attention_mask to SelfAttention
        x = self.ln1(x)
        x = x + self.attn(x, attention_mask=attention_mask)

        x = self.ln2(x)
        x = x + self.mlp(x)

        return x


class TransformerBlocks(nn.Module):
    def __init__(self, *modules: TransformerBlock):
        super().__init__()
        self.layers = list(modules)

    def __call__(
        self, x: mx.array, *, attention_mask: mx.array | None = None
    ) -> mx.array:
        for m in self.layers:
            x = m(x, attention_mask=attention_mask)
        return x


class NueLM(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()

        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embed)
        self.blocks = TransformerBlocks(
            *[TransformerBlock(config) for _ in range(config.n_layers)]
        )
        self.ln_f = nn.LayerNorm(config.n_embed)
        self.head = nn.Linear(config.n_embed, config.vocab_size, bias=False)

        # weight tying
        self.head.weight = self.tok_emb.weight

        # fp16
        self.set_dtype(mx.bfloat16)

        # 重みを初期化
        self.apply_to_modules(init_by_layer)
        mx.eval(self)

    def __call__(
        self,
        input_ids: mx.array,
        *,
        attention_mask: mx.array | None = None,
    ) -> mx.array:
        # input_ids: [B, T], attention_mask: [B, T]
        x = self.tok_emb(input_ids)  # [B, T, D]
        x = self.blocks(x, attention_mask=attention_mask)
        x = self.ln_f(x)
        logits = self.head(x)

        return logits


# NOTE: PyTorch の nn.init.trunc_normal_(module.weight, mean=0.0, std=0.02, a=-0.04, b=0.04)
#
# PyTorch は標準偏差 0.02 の正規分布での -0.04〜+0.04
# MLX の mx.random.truncated_normal は標準偏差 1 平均 0 の正規分布なので、切り取る範囲を
# 2.0 に設定することで、標準偏差 0.02 の正規分布での -0.04〜+0.04 に相当する
def trunc_normal_like(arr: mx.array, std=0.02, mean=0.0, trunc_scale=2.0) -> mx.array:
    """Return array shaped like `arr` from N(mean, std) truncated to ±trunc_scale·std."""
    # NOTE: fp32 で初期化。最後に cast することで、分布が崩れないようにする

    # 1) 標準正規の ±trunc_scale を切り取ってサンプリング
    sample = mx.random.truncated_normal(
        -trunc_scale, trunc_scale, shape=arr.shape, dtype=mx.float32
    )

    # 2) スケーリング & シフト
    return (sample * std + mean).astype(arr.dtype)


# 出力層重みを ±0.04 に制限することで、活性化後のスケールが 線形に 0.02 σ へ収束 -> 過大な logits を
# 抑制バイアスを 0、LayerNorm γ=1/β=0 とすることで 分散が早期に安定 → Cross-Entropy が発散しにくい
def init_by_layer(name: str, mod: nn.Module):
    if isinstance(mod, nn.Embedding):
        mod.apply(trunc_normal_like)
    elif isinstance(mod, nn.Linear):
        mod.apply(trunc_normal_like, filter_fn=lambda _, n, a: n == "weight")
        mod.apply(
            nn.init.constant(0.0, dtype=mx.bfloat16),
            filter_fn=lambda _, n, __: n == "bias",
        )
    elif isinstance(mod, nn.LayerNorm):
        # NOTE: 学習の安定化のためには、LayerNorm の weight/bias および Linear の bias は
        # float32 の方が望ましいが、ここではメモリ削減を優先して bfloat16 にする
        mod.apply(
            nn.init.constant(1.0, dtype=mx.bfloat16),
            filter_fn=lambda _, n, __: n == "weight",
        )
        mod.apply(
            nn.init.constant(0.0, dtype=mx.bfloat16),
            filter_fn=lambda _, n, __: n == "bias",
        )
