from abc import ABC

from nue.model.base import GPTConfig

from .base import TrainingOptions
from .tokenizer import TOKENIZER


class BaseTrainer(ABC):
    config: GPTConfig
    options: TrainingOptions

    def __init__(self, options: TrainingOptions):
        self.options = options
        self.config = GPTConfig(
            vocab_size=TOKENIZER.vocab_size(),
            ctx_len=options.ctx_len,
            n_embed=options.n_embed,
            n_heads=options.n_heads,
            n_layers=options.n_layers,
            mlp_ratio=options.mlp_ratio,
        )
