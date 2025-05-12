from abc import ABC, abstractmethod

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

    def train(
        self,
        *,
        log_validation_max_tokens: int = 50_000,
        measure_time: bool = False,
        override_base_lr: float | None = None,
    ) -> None:
        self._train(
            log_validation_max_tokens=log_validation_max_tokens,
            measure_time=measure_time,
            override_base_lr=override_base_lr,
        )

    @abstractmethod
    def _train(
        self,
        *,
        log_validation_max_tokens: int,
        measure_time: bool,
        override_base_lr: float | None,
    ) -> None:
        raise NotImplementedError
