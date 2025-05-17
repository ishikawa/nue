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

"""Training utilities built on top of mlx."""

import json
import math
import os
from functools import partial
from typing import Any, Callable, Iterator, Optional, cast

import click
import mlx.core as mx
import mlx.data
import mlx.nn as nn
from datasets import Dataset
from mlx.optimizers import AdamW, Optimizer

from nue.mlx.model import NueLM
from nue.model.base import GPTConfig
from nue.train.base import TrainingOptions
from nue.train.tokenizer import IGNORE_TOKEN_ID, PAD_TOKEN_ID
from nue.train.trainer import BaseTrainer


class MlxTrainer(BaseTrainer):
    model: NueLM

    train_stream: Any | None = None
    validation_stream: Any | None = None

    optimizer: Optimizer | None = None

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

    def on_train_initialize(self) -> None:
        pass

    def on_load_dataset(
        self,
        train_dataset: Dataset,
        validation_dataset: Dataset,
    ) -> None:
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
                    "input_ids",
                    dim=0,
                    size=self.options.ctx_len,
                    pad_value=PAD_TOKEN_ID,
                )
                .batch(self.options.batch_size)
            )

        # Load dataset into mlx buffer
        self.train_stream = hf_dataset_to_stream(train_dataset)
        self.validation_stream = hf_dataset_to_stream(validation_dataset)

    def on_train_prepare(self) -> None:
        lr_scheduler = get_cosine_schedule_with_warmup(
            base_lr=self.options.lr,
            num_warmup_steps=self.num_warmup_steps,
            num_training_steps=self.num_training_steps,
        )

        # NOTE: Adam だと速く収束するが鋭い谷に落ちやすい
        # optimizer = optim.Adam(model.parameters(), lr=lr)
        # NOTE: SGD は安定性を増すが、データ量が少ない時は収束しなかった
        optimizer = AdamW(
            learning_rate=lr_scheduler,
            # 過学習防止のため正則化
            weight_decay=0.01,
        )

        self.optimizer = optimizer

    def on_train_resume(
        self, checkpoint_path: str
    ) -> tuple[
        int,  # epoch
        int,  # step
    ]:
        # Load metadata
        meta_path = checkpoint_path.replace(".safetensors", ".meta.json")
        with open(meta_path, "r") as f:
            meta = json.load(f)
        # Load model weights
        self.model.load_weights(checkpoint_path)

        return meta["epoch"], meta["step"]

    @property
    def checkpoint_path(self) -> str:
        return os.path.join(self.options.model_dir, "checkpoint.safetensors")

    def save_checkpoint(self, *, epoch: int, step: int) -> None:
        assert self.model is not None

        self.model.save_weights(self.checkpoint_path)

        # Also save json metadata for convenience
        meta_path = self.checkpoint_path.replace(".safetensors", ".meta.json")
        with open(meta_path, "w") as f:
            json.dump({"epoch": epoch, "step": step}, f)

    def batch_train_iter(self) -> Iterator[dict[str, Any]]:
        assert self.train_stream is not None
        return iter(self.train_stream)

    def batch_validation_iter(self) -> Iterator[dict[str, Any]]:
        assert self.validation_stream is not None
        return iter(self.validation_stream)

    def synchronize_device(self) -> None:
        mx.synchronize()

    @property
    def learning_rate(self) -> float:
        assert self.optimizer is not None
        return float(self.optimizer.learning_rate)

    def _train(
        self,
        *,
        log_validation_max_tokens: int,
        measure_time: bool,
        override_base_lr: float | None,
    ) -> None:
        """Run the training loop.

        If ``override_base_lr`` is provided, the optimizer's learning rate is
        set to the given value before the loop starts. This mirrors the
        ``PyTorchTrainer`` behaviour where the optimizer's current learning
        rate is overwritten when resuming training.
        """
        assert self.model is not None
        assert self.train_dataset is not None
        assert self.validation_dataset is not None
        assert self.train_stream is not None
        assert self.validation_stream is not None
        assert self.optimizer is not None

        optimizer = self.optimizer

        # 学習開始前に学習率を変更する（学習再開時に上書きしたい場合）
        if override_base_lr is not None:
            optimizer.learning_rate = override_base_lr
            click.secho(
                f"Optimizer learning rate successfully set to: {override_base_lr}",
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

        # The state that will be captured as input and output
        captured_state = [self.model.state, optimizer.state]

        @partial(mx.compile, inputs=captured_state, outputs=captured_state)
        def model_step(
            input_ids: mx.array,
            labels: mx.array,
            attention_mask: mx.array | None = None,
        ) -> mx.array:
            loss_and_grad_fn = nn.value_and_grad(self.model, loss_fn)
            loss, grads = loss_and_grad_fn(
                self.model, input_ids, labels, attention_mask=attention_mask
            )

            # Update the model with the gradients. So far no computation has happened.
            optimizer.update(self.model, grads)

            return loss

        self.model.train()

        for iteration, batch in self.train_loop(
            log_validation_max_tokens=log_validation_max_tokens,
            measure_time=measure_time,
        ):
            with iteration.measure_io():
                batch = collate(batch, config=self.config)

            input_ids = batch["input_ids"]
            attn_mask = batch["attention_mask"]
            labels = batch["labels"]

            loss = model_step(
                input_ids,
                labels,
                attention_mask=attn_mask,
            )

            # Compute the new parameters but also the optimizer state.
            mx.eval(self.model.parameters(), optimizer.state)

            iteration.loss = float(loss)

            iteration.set_spinner_text()

    def generate(
        self,
        ids: list[int],
        *,
        max_new_tokens: int = 32,
        top_k: int | None = 20,
        temperature: float = 0.8,
        eos_id: int | None = None,
    ) -> list[int]:
        idx = mx.array([ids])  # [1, seq_len]

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.ctx_len :]
            logits = self.model(idx_cond)[:, -1, :]  # [1, V]

            if top_k is None or temperature <= 0.0:
                next_tok = logits.argmax(axis=-1, keepdims=True)
            else:
                # 大きい順に上位 k 個の logit を取得
                k = min(top_k, logits.shape[-1])
                topk_idx = mx.argpartition(logits, -k, axis=-1)[..., -k:]
                topk_logits = mx.take_along_axis(logits, topk_idx, axis=-1)

                # 温度スケーリングした logits をそのまま categorical へ
                sample_idx = mx.random.categorical(topk_logits / temperature, axis=-1)
                next_tok = mx.take_along_axis(topk_idx, sample_idx[..., None], axis=-1)

            idx = mx.concat([idx, next_tok], axis=1)

            if eos_id is not None and int(next_tok[0]) == eos_id:
                break

        return cast(list[int], idx[0].tolist())

    def evaluate(
        self,
        *,
        max_tokens: Optional[int] = None,
    ) -> float:
        assert self.model is not None
        assert self.validation_stream is not None

        total_loss = 0.0
        total_tokens = 0

        for batch in self.validation_stream:
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
        # NOTE: 損失計算は数値安定性の観点から `float32` で行う
        logits.astype(mx.float32),
        safe_labels,
        label_smoothing=label_smoothing,
        reduction="none",
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

    # --- pre-compute constant scalars (Python side, not traced) -------------
    warmup_steps_f = float(max(1, num_warmup_steps))
    decay_steps_f = float(max(1, num_training_steps - num_warmup_steps))
    two_pi_cycles = math.pi * 2.0 * num_cycles
    base_lr_arr = mx.array(base_lr, dtype=mx.float32)  # tensor scalar

    # ------------------------------------------------------------------------
    def schedule(step: mx.array) -> mx.array:
        step_f = step.astype(mx.float32)  # ensure fp tensor

        # linear warm-up (tensor)
        warmup_lr = (step_f / warmup_steps_f) * base_lr_arr

        # cosine decay (tensor)
        progress = (step_f - warmup_steps_f) / decay_steps_f
        cosine_mul = 0.5 * (1.0 + mx.cos(progress * two_pi_cycles))
        cosine_lr = mx.maximum(0.0, cosine_mul) * base_lr_arr

        # piece-wise select without Python if
        return mx.where(step_f < warmup_steps_f, warmup_lr, cosine_lr)

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
