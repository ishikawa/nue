import torch

from .base import GPTConfig
from .torch import MinimalGPT


def test_minimalgpt_forward_and_weights():
    cfg = GPTConfig(
        vocab_size=10,
        ctx_len=8,
        n_embed=16,
        n_heads=4,
        n_layers=2,
        mlp_ratio=4,
    )
    model = MinimalGPT(cfg)

    assert len(model.blocks) == cfg.n_layers

    input_ids = torch.zeros((2, cfg.ctx_len), dtype=torch.long)
    out = model(input_ids)

    assert out.shape == (2, cfg.ctx_len, cfg.vocab_size)
    assert model.head.weight.data_ptr() == model.tok_emb.weight.data_ptr()
