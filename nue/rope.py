import torch
import torch.nn as nn


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
