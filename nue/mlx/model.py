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
import mlx.core as mx
import mlx.nn as nn

from nue.model.base import GPTConfig


class Nue(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()

        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embed)
        self.blocks = [
            # TransformerBlock(config) for _ in range(config.n_layers)
        ]
        self.ln_f = nn.LayerNorm(config.n_embed)
        self.head = nn.Linear(config.n_embed, config.vocab_size, bias=False)
        # weight tying
        self.head.weight = self.tok_emb.weight

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
        for block in self.blocks:
            x = block(x, attention_mask=attention_mask)
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
    # 1) 標準正規の ±trunc_scale を切り取ってサンプリング
    sample = mx.random.truncated_normal(
        -trunc_scale, trunc_scale, shape=arr.shape, dtype=arr.dtype
    )

    # 2) スケーリング & シフト
    return sample * std + mean


# 出力層重みを ±0.04 に制限することで、活性化後のスケールが 線形に 0.02 σ へ収束 -> 過大な logits を抑制
# バイアスを 0、LayerNorm γ=1/β=0 とすることで 分散が早期に安定 → Cross-Entropy が発散しにくい
def init_by_layer(name: str, mod: nn.Module):
    if isinstance(mod, nn.Embedding):
        mod.apply(trunc_normal_like)
    elif isinstance(mod, nn.Linear):
        mod.apply(trunc_normal_like, filter_fn=lambda _, n, a: n == "weight")
        mod.apply(nn.init.constant(0.0), filter_fn=lambda _, n, __: n == "bias")
    elif isinstance(mod, nn.LayerNorm):
        mod.apply(nn.init.constant(1.0), filter_fn=lambda _, n, __: n == "weight")
        mod.apply(nn.init.constant(0.0), filter_fn=lambda _, n, __: n == "bias")
