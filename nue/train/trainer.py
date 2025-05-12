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
        click.secho("[1/7] Initialize", fg="green", bold=True)

        click.secho(
            f"vocab_size: {self.config.vocab_size}, device: {self.device_type}",
            fg="white",
        )

        # Save hyperparameters in JSON format
        with open(os.path.join(self.options.model_dir, "hparams.json"), "w") as f:
            json.dump(dataclasses.asdict(self.config), f, indent=4)

        self.on_train_initialize()

        # --------- 2) データセット準備 ---------
        click.secho("[2/7] Prepare dataset", fg="green", bold=True)

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

        # --------- 4) Optimizer & Scheduler ---------
        click.secho("[4/7] Prepare optimizer & scheduler", fg="green", bold=True)
        self.on_train_prepare()

        # --------- 5) 前回の学習状態を復元 ---------
        start_epoch = 0
        start_step = 0

        if os.path.exists(self.checkpoint_path):
            click.secho(
                f"[5/7] Resuming training from checkpoint {self.checkpoint_path}",
                fg="green",
                bold=True,
            )
            start_epoch, start_step = self.on_train_resume(
                checkpoint_path=self.checkpoint_path
            )
        else:
            click.secho("[5/7] Training from scratch", fg="bright_green", bold=True)
            os.makedirs(self.options.model_dir, exist_ok=True)

        self._train(
            start_epoch=start_epoch,
            start_step=start_step,
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
    def on_train_initialize(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def on_load_dataset(
        self,
        train_dataset: Dataset,
        validation_dataset: Dataset,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def on_train_prepare(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def on_train_resume(
        self, checkpoint_path: str
    ) -> tuple[
        int,  # epoch
        int,  # step
    ]:
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

    @property
    @abstractmethod
    def checkpoint_path(self) -> str:
        """
        The path to the checkpoint file. It must be the file path under the model directory.
        """
        raise NotImplementedError

    @abstractmethod
    def save_checkpoint(self, *, epoch: int, step: int) -> None:
        """
        Save the model checkpoint.

        Args:
            epoch (int): The current epoch.
            step (int): The current step.
        """
        raise NotImplementedError

    @abstractmethod
    def _train(
        self,
        start_epoch: int,
        start_step: int,
        *,
        log_validation_max_tokens: int,
        measure_time: bool,
        override_base_lr: float | None,
    ) -> None:
        raise NotImplementedError
