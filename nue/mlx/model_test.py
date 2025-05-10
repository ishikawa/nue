from nue.mlx.model import Nue
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
    m = Nue(config)
    assert m
    print(m.parameters())
