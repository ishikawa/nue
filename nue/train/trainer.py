import dataclasses
import json
import math
import os
from abc import ABC, abstractmethod

import click
from datasets import Dataset

from nue.model.base import GPTConfig
from nue.train.dataset import load_train_dataset
from nue.utils import format_number_abbrev

from .base import TrainingOptions
from .tokenizer import TOKENIZER


class BaseTrainer(ABC):
    config: GPTConfig
    options: TrainingOptions

    train_dataset: Dataset | None = None
    validation_dataset: Dataset | None = None

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

        # --------- 3) データセット準備 ---------
        click.secho("[3/7] Prepare dataset", fg="green", bold=True)

        dataset, total_tokens = load_train_dataset(
            ctx_len=self.options.ctx_len,
            chunk_overlap_len=self.options.chunk_overlap_len,
            override_data_size=self.options.override_data_size,
        )
        train_and_test_datasets = dataset.train_test_split(test_size=0.05)
        validation_dataset = train_and_test_datasets["test"]
        train_dataset = train_and_test_datasets["train"]

        click.secho(
            f"Total tokens: {format_number_abbrev(total_tokens)} ({total_tokens:,})",
            fg="cyan",
        )
        click.secho(
            f"Loader created (train: {len(train_dataset):,} rows, val: {len(validation_dataset):,} rows)",
            fg="cyan",
        )

        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset

        self.on_load_dataset(train_dataset, validation_dataset)

        click.secho(
            f"Estimated total steps: {self.num_training_steps}, Warmup steps: {self.num_warmup_steps}",
            fg="cyan",
        )

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
    def on_load_dataset(
        self,
        train_dataset: Dataset,
        validation_dataset: Dataset,
    ) -> None:
        raise NotImplementedError

    @property
    def num_training_steps_per_epoch(self) -> int:
        assert self.train_dataset is not None
        return math.ceil(len(self.train_dataset) / self.options.batch_size)

    @property
    def num_training_steps(self) -> int:
        return self.num_training_steps_per_epoch * self.options.n_epochs

    @property
    def num_warmup_steps(self) -> int:
        return int(min(self.num_training_steps * 0.05, self.options.max_warmup_steps))

    @abstractmethod
    def _train(
        self,
        *,
        log_validation_max_tokens: int,
        measure_time: bool,
        override_base_lr: float | None,
    ) -> None:
        raise NotImplementedError
