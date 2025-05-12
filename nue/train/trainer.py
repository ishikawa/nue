import dataclasses
import json
import os
from abc import ABC, abstractmethod

import click

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
        if self.options.seed is not None:
            self.manual_seed(self.options.seed)

        # --------- 1) Configuration ---------
        click.secho("[1/7] Configuration", fg="green", bold=True)

        click.secho(
            f"vocab_size: {self.config.vocab_size}, device: {self.device_type}",
            fg="cyan",
        )

        # Save hyperparameters in JSON format
        with open(os.path.join(self.options.model_dir, "hparams.json"), "w") as f:
            json.dump(dataclasses.asdict(self.config), f, indent=4)

        # --------- 2) Model 初期化 ---------
        click.secho("[2/7] Initialize model", fg="green", bold=True)
        self.initialize_model()

        self._train(
            log_validation_max_tokens=log_validation_max_tokens,
            measure_time=measure_time,
            override_base_lr=override_base_lr,
        )

    @abstractmethod
    def manual_seed(self, seed: int) -> None:
        raise NotImplementedError

    @property
    @abstractmethod
    def device_type(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def initialize_model(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def _train(
        self,
        *,
        log_validation_max_tokens: int,
        measure_time: bool,
        override_base_lr: float | None,
    ) -> None:
        raise NotImplementedError
