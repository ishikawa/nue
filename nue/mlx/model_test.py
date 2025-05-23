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

"""Tests for mlx GPT model implementation."""

import mlx.core as mx

from nue.mlx.model import NueLM
from nue.model.base import GPTConfig


def test_simple():
    config = GPTConfig(
        vocab_size=1000,
        ctx_len=128,
        n_embed=4,
        n_heads=2,
        n_layers=4,
        mlp_ratio=2,
    )
    m = NueLM(config)

    assert len(m.blocks.layers) == config.n_layers

    assert m.ln_f.weight.dtype == mx.bfloat16
    assert m.ln_f.bias.dtype == mx.bfloat16

    assert m.head.weight.dtype == mx.bfloat16

    assert m.blocks.layers[0].attn.causal.dtype == mx.bfloat16
