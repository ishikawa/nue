# Copyright 2025 Takanori Ishikawa
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import time
from typing import Any, Callable, Iterable, Iterator, Optional, cast

import click
import mlx.core as mx
import mlx.data
import mlx.nn as nn
import mlx.optimizers
import numpy as np
from datasets import Dataset
from termcolor import colored
from yaspin import yaspin

from nue.mlx.model import NueLM
from nue.model.base import GPTConfig
from nue.train.base import TrainingOptions
from nue.train.dataset import load_train_dataset
from nue.train.tokenizer import IGNORE_TOKEN_ID, PAD_TOKEN_ID, TOKENIZER
from nue.train.trainer import BaseTrainer
from nue.utils import format_number_abbrev


class MlxTrainer(BaseTrainer):
    model: NueLM

    def __init__(
        self,
        /,
        options: TrainingOptions,
    ) -> None:
        super().__init__(options)
        self.model = NueLM(self.config)

    def manual_seed(self, seed: int) -> None:
        mx.random.seed(seed)

    @property
    def device_type(self) -> str:
        match mx.default_device().type:
            case mx.DeviceType.cpu:
                return "cpu"
            case mx.DeviceType.gpu:
                return "gpu"
            case _:
                raise ValueError(f"Unknown device type: {mx.default_device().type}")

    def _train(
        self,
        *,
        log_validation_max_tokens: int,
        measure_time: bool,
        override_base_lr: float | None,
    ) -> None:
        options = self.options

        # --------- 3) データセット準備 ---------
        def build_hf_dataset_iter_fn(
            dataset: Dataset,
        ) -> Callable[[], Iterator[dict[str, Any]]]:
            def iter_fn() -> Iterator[dict[str, Any]]:
                for example in dataset:
                    yield {
                        "input_ids": example["input_ids"]  # type: ignore
                    }

            return iter_fn

        def hf_dataset_to_stream(dataset: Dataset) -> Any:
            return (
                mlx.data.stream_python_iterable(build_hf_dataset_iter_fn(dataset))  # type: ignore
                # Pad each sequence to the right
                .pad_to_size(
                    "input_ids", dim=0, size=options.ctx_len, pad_value=PAD_TOKEN_ID
                )
                .batch(options.batch_size)
            )

        click.secho("[3/7] Prepare dataset", fg="bright_green", bold=True)
        dataset, total_tokens = load_train_dataset(
            ctx_len=options.ctx_len,
            chunk_overlap_len=options.chunk_overlap_len,
            override_data_size=options.override_data_size,
        )
        click.secho(
            f"Total tokens: {format_number_abbrev(total_tokens)} ({total_tokens:,})",
            fg="cyan",
        )

        # dataset.set_format(type="numpy", columns=["input_ids"])

        # Split into train and validation datasets
        train_and_test_datasets = dataset.train_test_split(test_size=0.05)
        validation_dataset = train_and_test_datasets["test"]
        train_dataset = train_and_test_datasets["train"]

        click.secho(
            f"Loader created (train: {len(train_dataset):,} rows, val: {len(validation_dataset):,} rows)",
            fg="cyan",
        )

        # Load dataset into mlx buffer
        train_stream = hf_dataset_to_stream(train_dataset)
        validation_stream = hf_dataset_to_stream(validation_dataset)

        # --------- 4) Optimizer & Scheduler ---------
        click.secho("[4/7] Prepare optimizer & scheduler", fg="bright_green", bold=True)

        # --- スケジューラーの設定 ---
        # おおよその総学習ステップ数を計算 (エポック数 x 1エポックあたりのステップ数)
        # len(dataset) はチャンク化後の訓練データセットのサンプル数
        num_training_steps_per_epoch = math.ceil(
            len(train_dataset) / options.batch_size
        )
        num_training_steps = num_training_steps_per_epoch * options.n_epochs
        num_warmup_steps = int(min(num_training_steps * 0.05, options.max_warmup_steps))

        click.secho(
            f"Estimated total steps: {num_training_steps}, Warmup steps: {num_warmup_steps}",
            fg="cyan",
        )

        def loss_fn(
            model: nn.Module,
            input_ids: mx.array,
            labels: mx.array,
            attention_mask: mx.array | None = None,
        ) -> mx.array:
            logits = model(input_ids, attention_mask=attention_mask)
            return cross_entropy_mean(logits, labels)

        loss_and_grad_fn = nn.value_and_grad(self.model, loss_fn)

        lr_scheduler = get_cosine_schedule_with_warmup(
            base_lr=options.lr,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )

        # NOTE: Adam だと速く収束するが鋭い谷に落ちやすい
        # optimizer = optim.Adam(model.parameters(), lr=lr)
        # NOTE: SGD は安定性を増すが、データ量が少ない時は収束しなかった
        optimizer = mlx.optimizers.AdamW(
            learning_rate=lr_scheduler,
            # 過学習防止のため正則化
            weight_decay=0.01,
        )

        click.secho("[5/7] Training from scratch", fg="bright_green", bold=True)
        self.model.train()

        with yaspin().cyan as spinner:

            def set_spinner_text(
                i_epoch: int,
                i_step: int,
                *,
                lr: float,
                loss: Optional[float] = None,
            ):
                p = (i_step + 1) / num_training_steps_per_epoch
                spinner.text = (
                    colored(
                        f"Epoch {i_epoch + 1}/{options.n_epochs} Step {i_step + 1} ({p:.1%})",
                        color="cyan",
                    )
                    + " ("
                    + f"lr: {lr:.8f}"
                    + (f", loss: {loss:.3f}" if loss is not None else "")
                    + ")"
                )

            epoch_loss = 0.0
            total_loss = 0.0

            t0 = 0.0
            t1 = 0.0
            t2 = 0.0
            t3 = 0.0
            t4 = 0.0

            io_elapsed = 0.0
            forward_elapsed = 0.0
            backward_elapsed = 0.0
            optimizer_elapsed = 0.0
            step_elapsed = 0.0

            i_step = 0

            for i_epoch in range(0, options.n_epochs):
                loader_iter = iter(train_stream)

                while True:
                    if measure_time:
                        mx.synchronize()
                        t0 = time.perf_counter()

                    batch = next(loader_iter)

                    batch = collate(batch, config=self.config)

                    input_ids = batch["input_ids"]
                    attn_mask = batch["attention_mask"]
                    labels = batch["labels"]

                    if measure_time:
                        mx.synchronize()
                        t1 = time.perf_counter()

                    loss, grads = loss_and_grad_fn(
                        self.model, input_ids, labels, attention_mask=attn_mask
                    )

                    if measure_time:
                        mx.synchronize()
                        mx.eval(loss, grads)
                        t2 = time.perf_counter()

                    # Update the model with the gradients. So far no computation has happened.
                    optimizer.update(self.model, grads)

                    if measure_time:
                        mx.synchronize()
                        mx.eval(self.model.parameters())
                        t3 = time.perf_counter()

                    # Compute the new parameters but also the optimizer state.
                    mx.eval(self.model.parameters(), optimizer.state)

                    if measure_time:
                        mx.synchronize()
                        t4 = time.perf_counter()

                    set_spinner_text(
                        i_epoch=i_epoch,
                        i_step=i_step,
                        lr=float(lr_scheduler(mx.array(i_step))),
                        loss=float(loss),
                    )

                    total_loss += loss.item()
                    epoch_loss += loss.item()

                    io_elapsed += t1 - t0
                    forward_elapsed += t2 - t1
                    backward_elapsed += t3 - t2
                    optimizer_elapsed += t4 - t3
                    step_elapsed += t4 - t0

                    # Log training progress
                    if (i_step + 1) % options.log_interval == 0:
                        try:
                            self.model.eval()

                            # Evaluate on validation dataset
                            val_loss = self.evaluate(
                                validation_stream,
                                max_tokens=log_validation_max_tokens,
                            )

                            # Generate samples

                            for text in self.generate_samples():
                                spinner.write(
                                    colored(
                                        f"  SAMPLE: {text}",
                                        "yellow",
                                    )
                                )
                        finally:
                            self.model.train()

                        # 平均 loss を計算
                        avg_loss = total_loss / options.log_interval
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

                        current_lr = float(lr_scheduler(mx.array(i_step)))
                        progress += f"{colored('lr=', 'cyan')}{current_lr:.6f} "

                        if measure_time:
                            progress += "("
                            progress += f"{step_elapsed / options.log_interval:.3f}s "
                            progress += f"{colored('io=', 'cyan')}{io_elapsed / options.log_interval:.3f}s "
                            progress += f"{colored('forward=', 'cyan')}{forward_elapsed / options.log_interval:.3f}s "
                            progress += f"{colored('backward=', 'cyan')}{backward_elapsed / options.log_interval:.3f}s "
                            progress += f"{colored('optimizer=', 'cyan')}{optimizer_elapsed / options.log_interval:.3f}s"
                            progress += ")"

                        spinner.write(progress)

                        total_loss = 0.0
                        io_elapsed = 0.0
                        forward_elapsed = 0.0
                        backward_elapsed = 0.0
                        optimizer_elapsed = 0.0
                        step_elapsed = 0.0

                    i_step += 1

    def _generate(self, idx: mx.array, max_new_tokens: int = 32):
        """Greedy text generation (for demo)."""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.ctx_len :]
            logits = self.model(idx_cond)[:, -1, :]  # [B,vocab]
            next_tok = logits.argmax(axis=-1, keepdims=True)
            idx = mx.concat([idx, next_tok], axis=1)

        return idx

    def _generate_text(
        self,
        prompt: str,
        *,
        max_new_length: int,
    ) -> str:
        assert self.model is not None

        ids: list[int] = TOKENIZER.EncodeAsIds(prompt)
        idx = mx.array([ids], dtype=mx.int64)
        out = self._generate(idx, max_new_tokens=max_new_length)[0].tolist()

        return TOKENIZER.DecodeIds(out)

    def generate_samples(self) -> Iterable[str]:
        for prompt in ["富士山は", "Alan Turing is "]:
            yield self._generate_text(prompt, max_new_length=50)

    def evaluate(
        self,
        data_loader: Any,  # mlx.data.Stream
        *,
        max_tokens: Optional[int] = None,
    ) -> float:
        assert self.model is not None

        total_loss = 0.0
        total_tokens = 0

        for batch in data_loader:
            batch = collate(batch, config=self.config)

            input_ids = batch["input_ids"]
            attn_mask = batch["attention_mask"]
            labels = batch["labels"]

            logits = self.model(
                input_ids,
                attention_mask=attn_mask,
            )

            loss = cross_entropy_mean(
                logits,
                labels,
            )

            num_tokens = cast(mx.array, (labels != IGNORE_TOKEN_ID)).sum()
            total_loss += loss * num_tokens
            total_tokens += num_tokens

            if max_tokens is not None and total_tokens >= max_tokens:
                break

        avg_loss = total_loss / total_tokens
        return float(avg_loss)


def cross_entropy_mean(
    logits: mx.array,  # (B, T, V)
    labels: mx.array,  # (B, T)
    label_smoothing: float = 0.0,
) -> mx.array:
    """
    Compute the cross-entropy loss mean. Ignores IGNORE_TOKEN_ID.
    """
    vocab_size = logits.shape[-1]

    # logits と labels の shape を揃える
    logits = logits.reshape(-1, vocab_size)  # shape = (B*T, V)
    labels = labels.reshape(-1)  # shape = (B*T,)

    # IGNORE_TOKEN_ID を損失計算から除外するために、
    # IGNORE_TOKEN_ID の部分を 0 にする
    mask = cast(mx.array, labels != IGNORE_TOKEN_ID)
    safe_labels = mx.where(mask, labels, 0)

    # 各トークンごとの loss 値
    # shape: (B*T,)
    per_token_loss = nn.losses.cross_entropy(
        logits, safe_labels, label_smoothing=label_smoothing, reduction="none"
    )

    # mask で無視する部分を考慮しつつ平均を取る
    per_token_loss = per_token_loss * mask.astype(per_token_loss.dtype)
    return mx.sum(per_token_loss) / mx.maximum(mx.sum(mask), 1)


def get_cosine_schedule_with_warmup(
    base_lr: float,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
) -> Callable[[mx.array], mx.array]:
    """
    Create a learning rate schedule that linearly increases the learning rate from
    0.0 to lr over ``num_warmup_steps``, then decreases to 0.0 on a cosine schedule over
    the remaining ``num_training_steps-num_warmup_steps`` (assuming ``num_cycles`` = 0.5).

    This is based on the Hugging Face implementation
    https://github.com/huggingface/transformers/blob/v4.23.1/src/transformers/optimization.py#L104.

    Args:
        num_warmup_steps (int): The number of steps for the warmup phase.
        num_training_steps (int): The total number of training steps.
        num_cycles (float): The number of waves in the cosine schedule. Defaults to 0.5
            (decrease from the max value to 0 following a half-cosine).
        last_epoch (int): The index of the last epoch when resuming training. Defaults to -1

    """

    def schedule(current_step: mx.array) -> mx.array:
        # linear warmup phase
        if current_step < num_warmup_steps:
            return current_step / max(1, num_warmup_steps) * base_lr

        # cosine
        progress = (current_step - num_warmup_steps) / max(
            1, num_training_steps - num_warmup_steps
        )

        cosine_lr_multiple = 0.5 * (
            1.0 + math.cos(math.pi * num_cycles * 2.0 * progress)
        )
        return mx.maximum(mx.array(0.0), cosine_lr_multiple) * base_lr

    return schedule


def collate(batch: dict[str, Any], *, config: GPTConfig) -> dict[str, mx.array]:
    input_ids = mx.array(batch["input_ids"])

    # 2) Build attention mask (boolean mask)
    attn_mask = mx.array(input_ids != PAD_TOKEN_ID)

    # 3) Create labels
    # - 次トークン予測タスクでは、labels[i] が input_ids[i+1] に対応
    # - パディング部分は損失計算から除外する必要がある

    # batch と同じ shape で全ての要素を IGNORE (損失計算で無視される値) で初期化
    labels = mx.full(input_ids.shape, IGNORE_TOKEN_ID)

    # input_ids を左シフトして labels に代入することで、
    # labels[i] <- input_ids[i+1]
    labels[:, :-1] = input_ids[:, 1:]

    # labels の要素で PAD_ID となっている箇所を IGNORE にする
    labels = mx.where(labels == PAD_TOKEN_ID, IGNORE_TOKEN_ID, labels)

    return {
        "input_ids": input_ids,
        "attention_mask": attn_mask,
        "labels": labels,
    }
