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

import dataclasses
import json
import math
import os
from typing import Any, Callable, Iterable, Optional, cast

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
from nue.train.base import TrainingOptions, TrainingSession
from nue.train.dataset import load_train_dataset
from nue.train.tokenizer import IGNORE_TOKEN_ID, PAD_TOKEN_ID, TOKENIZER
from nue.utils import format_number_abbrev


class MlxTrainer:
    config: GPTConfig
    options: TrainingOptions
    model: NueLM

    def __init__(
        self,
        /,
        options: TrainingOptions,
    ) -> None:
        self.options = options
        self.config = GPTConfig(
            vocab_size=TOKENIZER.vocab_size(),
            ctx_len=options.ctx_len,
            n_embed=options.n_embed,
            n_heads=options.n_heads,
            n_layers=options.n_layers,
            mlp_ratio=options.mlp_ratio,
        )
        self.model = NueLM(self.config)

    def train(
        self,
        session: TrainingSession,
        *,
        log_validation_max_tokens: int = 50_000,
        measure_time: bool = False,
        override_base_lr: float | None = None,
    ) -> None:
        options = self.options

        # シード設定
        if options.seed is not None:
            mx.random.seed(options.seed)

        # --------- 1) Configuration ---------
        click.secho("[1/7] Configuration", fg="green", bold=True)

        click.secho(f"vocab_size: {self.config.vocab_size}", fg="cyan")

        # Save hyperparameters in JSON format
        with open(os.path.join(options.model_dir, "hparams.json"), "w") as f:
            json.dump(dataclasses.asdict(self.config), f, indent=4)

        # --------- 2) Initialize model ---------
        click.secho("[2/7] Initialize model", fg="green", bold=True)

        # --------- 3) データセット準備 ---------
        def hf_dataset_to_buffer(dataset: Dataset) -> Any:
            dicts = []
            for input_ids in dataset["input_ids"]:
                dicts.append({"input_ids": input_ids})

            assert isinstance(dicts, list)
            assert isinstance(dicts[0], dict)
            assert isinstance(dicts[0]["input_ids"], np.ndarray)

            return mlx.data.buffer_from_vector(dicts)  # type: ignore

        def buffer_to_stream(buffer: Any) -> Any:
            return (
                buffer.to_stream()
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

        dataset.set_format(type="numpy", columns=["input_ids"])

        # Split into train and validation datasets
        train_and_test_datasets = dataset.train_test_split(test_size=0.05)
        validation_dataset = train_and_test_datasets["test"]
        train_dataset = train_and_test_datasets["train"]

        # Load dataset into mlx buffer
        train_buffer = hf_dataset_to_buffer(train_dataset)
        validation_buffer = hf_dataset_to_buffer(validation_dataset)

        click.secho(
            f"Loader created (train: {len(train_buffer):,} rows, val: {len(validation_buffer):,} rows)",
            fg="cyan",
        )

        train_stream = buffer_to_stream(train_buffer)
        validation_stream = buffer_to_stream(validation_buffer)

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

        loss_and_grad_fn = nn.value_and_grad(self.model, cross_entropy_mean)

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
                logits_mean: Optional[float] = None,
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
                    + (
                        f", logits mean: {logits_mean:.3f}"
                        if logits_mean is not None
                        else ""
                    )
                    + ")"
                )

            total_loss = 0.0
            epoch_loss = 0.0

            i_step = 0

            for i_epoch in range(0, options.n_epochs):
                for input_ids in train_stream:
                    input_ids = mx.array(input_ids["input_ids"])

                    # 1) Build attention mask (boolean mask)
                    attn_mask = mx.array(input_ids != PAD_TOKEN_ID)

                    # 2) Create labels
                    # - 次トークン予測タスクでは、labels[i] が input_ids[i+1] に対応
                    # - パディング部分は損失計算から除外する必要がある

                    # batch と同じ shape で全ての要素を IGNORE (損失計算で無視される値) で初期化
                    labels = mx.full(input_ids.shape, IGNORE_TOKEN_ID)

                    # input_ids を左シフトして labels に代入することで、
                    # labels[i] <- input_ids[i+1]
                    labels[:, :-1] = input_ids[:, 1:]

                    # labels の要素で PAD_ID となっている箇所を IGNORE にする
                    labels = mx.where(labels == PAD_TOKEN_ID, IGNORE_TOKEN_ID, labels)

                    mx.eval(input_ids)
                    mx.eval(attn_mask)
                    mx.eval(labels)

                    logits = self.model(input_ids, attention_mask=attn_mask)
                    loss, grads = loss_and_grad_fn(logits, labels)

                    # Update the model with the gradients. So far no computation has happened.
                    optimizer.update(self.model, grads)

                    # Compute the new parameters but also the optimizer state.
                    mx.eval(self.model.parameters(), optimizer.state)

                    logits_mean = float(logits.abs().mean())
                    current_lr = float(lr_scheduler(mx.array(i_step)))

                    set_spinner_text(
                        i_epoch=i_epoch,
                        i_step=i_step,
                        lr=current_lr,
                        loss=float(loss),
                        logits_mean=logits_mean,
                    )

                    total_loss += loss.item()
                    epoch_loss += loss.item()

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

                        spinner.write(progress)

                        total_loss = 0.0

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
        data_stream: Any,  # mlx.data.Stream
        *,
        max_tokens: Optional[int] = None,
    ) -> float:
        assert self.model is not None

        total_loss = 0.0
        total_tokens = 0

        for batch in data_stream:
            input_ids = mx.array(batch["input_ids"])

            # 1) Build attention mask (boolean mask)
            attn_mask = mx.array(input_ids != PAD_TOKEN_ID)

            # 2) Create labels
            # - 次トークン予測タスクでは、labels[i] が input_ids[i+1] に対応
            # - パディング部分は損失計算から除外する必要がある

            # batch と同じ shape で全ての要素を IGNORE (損失計算で無視される値) で初期化
            labels = mx.full(input_ids.shape, IGNORE_TOKEN_ID)

            # input_ids を左シフトして labels に代入することで、
            # labels[i] <- input_ids[i+1]
            labels[:, :-1] = input_ids[:, 1:]

            # labels の要素で PAD_ID となっている箇所を IGNORE にする
            labels = mx.where(labels == PAD_TOKEN_ID, IGNORE_TOKEN_ID, labels)

            mx.eval(input_ids)
            mx.eval(attn_mask)
            mx.eval(labels)

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
