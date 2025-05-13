import dataclasses
import json
import math
import os
import time
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Any, Generator, Iterable, Iterator, Optional

import click
from datasets import Dataset
from termcolor import colored
from yaspin import yaspin
from yaspin.core import Yaspin

from nue.model.base import GPTConfig
from nue.train.dataset import load_train_dataset
from nue.utils import format_number_abbrev

from .base import TrainingOptions
from .tokenizer import TOKENIZER


class TrainingIteration:
    _trainer: "BaseTrainer"

    spinner: Yaspin

    i_epoch: int
    i_step: int
    measure_time: bool

    io_elapsed: float = 0.0
    forward_elapsed: float = 0.0
    backward_elapsed: float = 0.0
    optimizer_elapsed: float = 0.0
    step_elapsed: float = 0.0

    loss: float | None = None

    def __init__(
        self,
        trainer: "BaseTrainer",
        *,
        spinner: Yaspin,
        i_epoch: int,
        i_step: int,
        measure_time: bool,
    ):
        self.spinner = spinner
        self._trainer = trainer
        self.i_epoch = i_epoch
        self.i_step = i_step
        self.measure_time = measure_time

    def set_spinner_text(
        self,
        *,
        logits_mean: Optional[float] = None,
    ):
        lr = self._trainer.learning_rate
        progress = (self.i_step + 1) / self._trainer.num_training_steps_per_epoch

        self.spinner.text = (
            colored(
                f"Epoch {self.i_epoch + 1}/{self._trainer.options.n_epochs} Step {self.i_step + 1} ({progress:.1%})",
                color="cyan",
            )
            + " ("
            + f"lr: {lr:.8f}"
            + (f", loss: {self.loss:.3f}" if self.loss is not None else "")
            + (f", logits mean: {logits_mean:.3f}" if logits_mean is not None else "")
            + ")"
        )

    @contextmanager
    def _measure(self, name: str):
        t = 0

        if self.measure_time:
            self._trainer.synchronize_device()
            t = time.perf_counter()

        try:
            yield
        finally:
            if self.measure_time:
                self._trainer.synchronize_device()
                m = getattr(self, name)
                setattr(self, name, m + time.perf_counter() - t)

    def measure_io(self):
        return self._measure("io_elapsed")

    def measure_forward(self):
        return self._measure("forward_elapsed")

    def measure_backward(self):
        return self._measure("backward_elapsed")

    def measure_optimizer(self):
        return self._measure("optimizer_elapsed")

    def measure_step(self):
        return self._measure("step_elapsed")


class BaseTrainer(ABC):
    config: GPTConfig
    options: TrainingOptions

    train_dataset: Dataset | None = None
    validation_dataset: Dataset | None = None

    start_epoch: int = 0
    start_step: int = 0

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
        self.start_epoch = 0
        self.start_step = 0

        if os.path.exists(self.checkpoint_path):
            click.secho(
                f"[5/7] Resuming training from checkpoint {self.checkpoint_path}",
                fg="green",
                bold=True,
            )
            self.start_epoch, self.start_step = self.on_train_resume(
                checkpoint_path=self.checkpoint_path
            )
        else:
            click.secho("[5/7] Training from scratch", fg="bright_green", bold=True)
            os.makedirs(self.options.model_dir, exist_ok=True)

        click.secho("[6/7] Start training loop", fg="green", bold=True)
        self._train(
            log_validation_max_tokens=log_validation_max_tokens,
            measure_time=measure_time,
            override_base_lr=override_base_lr,
        )

    def _generate_text(
        self,
        prompt: str,
        *,
        max_new_length: int,
    ) -> str:
        ids: list[int] = TOKENIZER.EncodeAsIds(prompt)
        out = self.generate(ids, max_new_tokens=max_new_length)

        return TOKENIZER.DecodeIds(out)

    def _generate_samples(self) -> Iterable[str]:
        for prompt in ["富士山は", "Alan Turing is "]:
            yield self._generate_text(prompt, max_new_length=50)

    # --- Invoked by subclass ---

    def train_loop(
        self,
        *,
        log_validation_max_tokens: int,
        measure_time: bool = False,
    ) -> Generator[tuple[TrainingIteration, dict[str, Any]], None, None]:
        with yaspin().cyan as spinner:
            for i_epoch in range(self.start_epoch, self.options.n_epochs):
                epoch_loss = 0.0
                total_loss = 0.0

                io_elapsed = 0.0
                forward_elapsed = 0.0
                backward_elapsed = 0.0
                optimizer_elapsed = 0.0
                step_elapsed = 0.0

                i_step = 0
                loader_iter = self.batch_train_iter()

                while True:
                    try:
                        # 学習再開時には、開始前のデータをスキップする
                        if i_epoch == self.start_epoch and i_step < self.start_step:
                            next(loader_iter)
                            continue

                        it = TrainingIteration(
                            trainer=self,
                            spinner=spinner,
                            i_epoch=i_epoch,
                            i_step=i_step,
                            measure_time=measure_time,
                        )

                        with it.measure_step():
                            with it.measure_io():
                                batch = next(loader_iter)

                            yield it, batch

                        # Complete step
                        assert it.loss is not None

                        total_loss += it.loss
                        epoch_loss += it.loss

                        io_elapsed += it.io_elapsed
                        forward_elapsed += it.forward_elapsed
                        backward_elapsed += it.backward_elapsed
                        optimizer_elapsed += it.optimizer_elapsed
                        step_elapsed += it.step_elapsed

                        # Log training progress
                        if (i_step + 1) % self.options.log_interval == 0:
                            # Evaluate on validation dataset
                            val_loss = self.evaluate(
                                max_tokens=log_validation_max_tokens,
                            )

                            # Generate samples
                            for text in self._generate_samples():
                                spinner.write(
                                    colored(
                                        f"  SAMPLE: {text}",
                                        "yellow",
                                    )
                                )

                            # 平均 loss を計算
                            avg_loss = total_loss / self.options.log_interval
                            # perplexity
                            ppl = math.exp(avg_loss)
                            # validation perplexity
                            val_ppl = math.exp(val_loss)

                            progress = (
                                f"  Step {i_step + 1} "
                                + f"{colored('loss=', 'cyan')}{avg_loss:.3f} "
                                + f"{colored('ppl=', 'cyan')}{ppl:.3f} "
                                + f"{colored('val_loss=', 'cyan')}{val_loss:.3f} "
                                + f"{colored('val_ppl=', 'cyan')}{val_ppl:.3f} "
                            )

                            progress += (
                                f"{colored('lr=', 'cyan')}{self.learning_rate:.6f} "
                            )

                            if measure_time:
                                progress += "("
                                progress += (
                                    f"{step_elapsed / self.options.log_interval:.3f}s "
                                )
                                progress += f"{colored('io=', 'cyan')}{io_elapsed / self.options.log_interval:.3f}s "
                                progress += f"{colored('forward=', 'cyan')}{forward_elapsed / self.options.log_interval:.3f}s "
                                progress += f"{colored('backward=', 'cyan')}{backward_elapsed / self.options.log_interval:.3f}s "
                                progress += f"{colored('optimizer=', 'cyan')}{optimizer_elapsed / self.options.log_interval:.3f}s"
                                progress += ")"

                            spinner.write(progress)

                            total_loss = 0.0
                            io_elapsed = 0.0
                            forward_elapsed = 0.0
                            backward_elapsed = 0.0
                            optimizer_elapsed = 0.0
                            step_elapsed = 0.0

                        # Save model checkpoint
                        if (i_step + 1) % self.options.save_interval == 0:
                            # Epoch is not finished yet, so -1
                            self.save_checkpoint(epoch=i_epoch - 1, step=i_step)

                    except StopIteration:
                        break
                    finally:
                        i_step += 1

                # エポック終わりのサンプル生成と評価
                try:
                    # 評価
                    val_loss = self.evaluate()

                    # サンプル生成
                    for text in self._generate_samples():
                        spinner.write(
                            colored(
                                f"  Sample generation: {text}",
                                "yellow",
                            )
                        )
                finally:
                    pass

                avg_epoch_loss = epoch_loss / i_step

                spinner.write(
                    colored(
                        f"Epoch {i_epoch + 1}/{self.options.n_epochs} finished",
                        "magenta",
                        attrs=["bold"],
                    )
                    + colored(
                        f" (avg loss={avg_epoch_loss:.4f}, val loss={val_loss:.4f})",
                        "magenta",
                    )
                )

                # Save epoch checkpoint
                click.secho(
                    f"Saving epoch checkpoint at {self.checkpoint_path}",
                    fg="magenta",
                    bold=True,
                )
                self.save_checkpoint(epoch=i_epoch, step=0)

            # --------- モデル保存 ---------
            click.secho(
                f"[7/7] Saving model to {self.checkpoint_path}...",
                fg="bright_green",
                bold=True,
            )

            self.save_checkpoint(epoch=self.options.n_epochs, step=0)

    # --- Override by subclass ---

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
    def learning_rate(self) -> float:
        raise NotImplementedError

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
    def batch_train_iter(self) -> Iterator[dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def batch_validation_iter(self) -> Iterator[dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def synchronize_device(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def evaluate(
        self,
        *,
        max_tokens: int | None = None,
    ) -> float:
        raise NotImplementedError

    @abstractmethod
    def generate(self, ids: list[int], *, max_new_tokens: int = 32) -> list[int]:
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
