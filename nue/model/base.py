from dataclasses import dataclass


@dataclass(frozen=True)
class GPTConfig:
    vocab_size: int  # BPE vocab (same as GPT‑2)
    ctx_len: int  # maximum sequence length
    n_embed: int  # model dimension (d_model)
    n_heads: int  # number of attention heads
    n_layers: int  # transformer blocks
    mlp_ratio: int  # feed‑forward expansion ratio
