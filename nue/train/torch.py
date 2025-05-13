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
import os
import platform
import random
from contextlib import contextmanager
from functools import partial
from typing import Any, Iterator, Optional, cast

import click
import torch
from datasets import Dataset

# from torch.amp.grad_scaler import GradScaler
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from nue.model.torch import MinimalGPT, init_weights
from nue.train.trainer import BaseTrainer

from .base import TrainingOptions
from .tokenizer import IGNORE_TOKEN_ID, PAD_TOKEN_ID

# NOTE: torch >= 2.6.0 かつ MPS Backend だと SDPA で NaN が出る
#
# [MPS] MultiheadAttention with masks and dropout produces NaNs #151667
# https://github.com/pytorch/pytorch/issues/151667
#
# このフラグを True にすると、MPS での評価時は CPU で推論する
CPU_EVALUATION_ON_MPS_BACKEND = True

# Platform detection
PLATFORM_MAC = "Darwin" in platform.system()
PLATFORM_WINDOWS = "Windows" in platform.system()

# Device check
if torch.cuda.is_available():
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("`bfloat16` is not supported on this device")


def collate_pad(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    # 1) 各シーケンス
    seqs = [b["input_ids"] for b in batch]

    # 2) 右パディング
    padded = pad_sequence(seqs, batch_first=True, padding_value=PAD_TOKEN_ID)  # [B, T]

    # 3) attention_mask （1=実トークン、0=PAD）
    attn_mask = (padded != PAD_TOKEN_ID).long()  # [B, T]

    # 4) Labels の作成
    # - 次トークン予測タスクでは、labels[i] が input_ids[i+1] に対応
    # - パディング部分は損失計算から除外する必要がある

    # 全ての要素を IGNORE (損失計算で無視される値) で初期化
    labels = torch.full_like(padded, IGNORE_TOKEN_ID)

    # input_ids を左シフトして labels に代入することで、
    # labels[i] <- input_ids[i+1]
    labels[:, :-1] = padded[:, 1:]

    # ラベルが PAD_ID となっている箇所を IGNORE にする
    labels[labels == PAD_TOKEN_ID] = IGNORE_TOKEN_ID

    return {
        "input_ids": padded,
        "attention_mask": attn_mask,
        "labels": labels,
    }


class PyTorchTrainer(BaseTrainer):
    model: torch.nn.Module | None = None
    device: torch.device

    train_loader: DataLoader
    validation_loader: DataLoader

    optimizer: torch.optim.Optimizer | None = None
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None
    criterion: torch.nn.Module | None = None

    def __init__(
        self,
        /,
        options: TrainingOptions,
    ) -> None:
        super().__init__(options)
        self.device = detect_device()

    def manual_seed(self, seed: int) -> None:
        torch.manual_seed(seed)

    @property
    def device_type(self) -> str:
        return self.device.type

    def on_train_initialize(self) -> None:
        model = MinimalGPT(self.config).to(torch.bfloat16).to(self.device)
        model.apply(init_weights)

        # Compile model
        if self.device.type != "mps":
            model = cast(torch.nn.Module, torch.compile(model, mode="max-autotune"))

        self.model = model

    def on_load_dataset(
        self,
        train_dataset: Dataset,
        validation_dataset: Dataset,
    ) -> None:
        train_dataset.set_format(type="torch", columns=["input_ids"])
        validation_dataset.set_format(type="torch", columns=["input_ids"])

        # DataLoader
        # - train_loader のみ並列化とメモリ固定
        # - macOS/Windows では並列化しない
        if PLATFORM_MAC or PLATFORM_WINDOWS:
            data_loader_num_workers = 0
        else:
            data_loader_num_workers = os.cpu_count() or 0

        train_loader = DataLoader(
            train_dataset,  # type: ignore
            batch_size=self.options.batch_size,
            shuffle=True,
            collate_fn=collate_pad,
            num_workers=data_loader_num_workers,
            persistent_workers=data_loader_num_workers > 0,
            pin_memory=self.device.type != "mps",
        )
        validation_loader = DataLoader(
            validation_dataset,  # type: ignore
            batch_size=self.options.batch_size,
            shuffle=False,
            collate_fn=collate_pad,
        )

        self.train_loader = train_loader
        self.validation_loader = validation_loader

    def on_train_prepare(self) -> None:
        assert self.model is not None

        # NOTE: Adam だと速く収束するが鋭い谷に落ちやすい
        # optimizer = optim.Adam(model.parameters(), lr=lr)
        # NOTE: SGD は安定性を増すが、データ量が少ない時は収束しなかった
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.options.lr,
            # 過学習防止のため正則化
            weight_decay=0.01,
            fused=True,
        )

        # --- スケジューラーの設定 ---
        # おおよその総学習ステップ数を計算 (エポック数 x 1エポックあたりのステップ数)
        # len(dataset) はチャンク化後の訓練データセットのサンプル数

        # 学習率スケジューラー: 線形ウォームアップ後、トレーニング終了まで余弦減衰
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.num_warmup_steps,
            num_training_steps=self.num_training_steps,
        )

        self.criterion = torch.nn.CrossEntropyLoss(
            reduction="mean",
            # Label smoothing for better generalization
            label_smoothing=0.1,
            # Padding token ID
            ignore_index=IGNORE_TOKEN_ID,
        )

    def on_train_resume(
        self, checkpoint_path: str
    ) -> tuple[
        int,  # epoch
        int,  # step
    ]:
        assert self.model is not None
        assert self.optimizer is not None
        assert self.scheduler is not None

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state"])

        last_epoch = int(checkpoint["epoch"])
        last_step = int(checkpoint["step"])

        start_epoch = last_epoch + 1
        start_step = last_step + 1

        self.scheduler.last_epoch = last_step

        return start_epoch, start_step

    @property
    def checkpoint_path(self) -> str:
        return os.path.join(self.options.model_dir, "checkpoint.pt")

    def save_checkpoint(self, *, epoch: int, step: int) -> None:
        assert self.model is not None
        assert self.optimizer is not None
        assert self.scheduler is not None

        torch.save(
            {
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "scheduler_state": self.scheduler.state_dict(),
                "epoch": epoch,
                "step": step,
            },
            self.checkpoint_path,
        )

    def batch_train_iter(self) -> Iterator[dict[str, Any]]:
        return iter(self.train_loader)

    def batch_validation_iter(self) -> Iterator[dict[str, Any]]:
        return iter(self.validation_loader)

    def synchronize_device(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        elif self.device.type == "mps":
            torch.mps.synchronize()

    @property
    def learning_rate(self) -> float:
        assert self.scheduler is not None
        return self.scheduler.get_last_lr()[0]

    def _train(
        self,
        *,
        log_validation_max_tokens: int,
        measure_time: bool,
        override_base_lr: float | None,
    ) -> None:
        assert self.model is not None
        assert self.train_dataset is not None
        assert self.validation_dataset is not None
        assert self.train_loader is not None
        assert self.validation_loader is not None
        assert self.optimizer is not None
        assert self.scheduler is not None
        assert self.criterion is not None

        model = self.model
        optimizer = self.optimizer
        scheduler = self.scheduler
        criterion = self.criterion

        # 学習開始前に学習率を変更する（学習再開時に上書きしたい場合）
        if override_base_lr is not None:
            for param_group in optimizer.param_groups:
                param_group["lr"] = override_base_lr

            # スケジューラの base_lrs を変更
            # LambdaLR の場合、base_lrs はリストなので、各要素を変更
            for i in range(len(scheduler.base_lrs)):
                scheduler.base_lrs[i] = override_base_lr
            click.secho(
                f"Optimizer and Scheduler base_lrs successfully set to: {override_base_lr}",
                fg="cyan",
            )

        model.train()

        # MPS では AMP (自動混合精度) を使用しない
        use_amp = self.device.type != "mps"

        # bfloat16 は十分な精度を持つので、GradScaler 不要
        # grad_scaler = GradScaler(self.device.type, enabled=use_amp)

        for iteration, batch in self.train_loop(
            log_validation_max_tokens=log_validation_max_tokens,
            measure_time=measure_time,
        ):
            # Runs the forward pass under `autocast`.
            with torch.autocast(
                device_type=self.device.type,
                dtype=torch.bfloat16,
                enabled=use_amp,
            ):
                with iteration.measure_forward():
                    input_ids = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)
                    labels = batch["labels"].to(self.device)

                    logits = model(
                        input_ids,
                        attention_mask=attention_mask,
                    )

                loss = criterion(
                    logits.view(-1, self.config.vocab_size),
                    labels.view(-1),
                )
                iteration.loss = float(loss.item())

            # Exits `autocast` before backward().
            logits_mean = logits.abs().mean().item()

            iteration.set_spinner_text(
                logits_mean=logits_mean,
            )

            with iteration.measure_backward():
                # 勾配の計算
                # grad_scaler.scale(loss).backward()
                loss.backward()

            # 勾配のクリッピング
            # ただし、小規模モデルは毎 step でなくても安定する
            if iteration.i_step % 4 == 0:
                # Un-scales the gradients of optimizer's assigned parameters in-place
                # grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            with iteration.measure_optimizer():
                # パラメータの更新
                # grad_scaler.step(optimizer)
                # grad_scaler.update()
                optimizer.step()
                scheduler.step()

                # 勾配の初期化
                optimizer.zero_grad()

    @torch.no_grad()
    def generate(
        self,
        ids: list[int],
        *,
        max_new_tokens: int = 32,
        top_k: Optional[int] = 20,
        temperature: float = 0.8,
    ) -> list[int]:
        assert self.model is not None

        with self.use_cpu_on_mps():
            idx = torch.tensor([ids], dtype=torch.long).to(self.device)

            for _ in range(max_new_tokens):
                idx_cond = idx[:, -self.config.ctx_len :]
                logits = self.model(idx_cond)[:, -1, :]  # [B, vocab]

                if top_k is None:
                    # Greedy
                    next_tok = logits.argmax(dim=-1, keepdim=True)
                else:
                    # 1) 温度スケーリング
                    scaled = logits / temperature

                    # 2) Softmax で確率に変換
                    probs = torch.softmax(scaled, dim=-1)  # [1, vocab]

                    # 3) 上位 top_k を選択
                    topk_probs, topk_indices = torch.topk(
                        probs, top_k, dim=-1
                    )  # [1, k]

                    # 4) Python で重み付きサンプリング
                    probs_list = topk_probs[0].tolist()
                    indices_list = topk_indices[0].tolist()
                    sampled = random.choices(indices_list, weights=probs_list, k=1)[0]

                    next_tok = torch.tensor(
                        [[sampled]], dtype=torch.long, device=self.device
                    )

                # 5) 連結
                idx = torch.cat([idx, next_tok], dim=1)

        return idx[0].cpu().tolist()

    def evaluate(
        self,
        *,
        max_tokens: Optional[int] = None,
    ) -> float:
        assert self.model is not None
        assert self.validation_loader is not None
        assert self.criterion is not None

        total_loss = 0.0
        total_tokens = 0

        self.model.eval()
        try:
            with torch.no_grad():
                with self.use_cpu_on_mps():
                    for batch in self.validation_loader:
                        input_ids = batch["input_ids"].to(self.device)
                        attention_mask = batch["attention_mask"].to(self.device)
                        labels = batch["labels"].to(self.device)

                        logits = self.model(
                            input_ids,
                            attention_mask=attention_mask,
                        )

                        loss = self.criterion(
                            logits.view(-1, self.config.vocab_size),
                            labels.view(-1),
                        )

                        num_tokens = (labels != IGNORE_TOKEN_ID).sum().item()
                        total_loss += loss.item() * num_tokens
                        total_tokens += num_tokens

                        if max_tokens is not None and total_tokens >= max_tokens:
                            break
        finally:
            self.model.train()

        avg_loss = total_loss / total_tokens
        return avg_loss

    @contextmanager
    def use_cpu_on_mps(self):
        """
        MPS backend のバグを回避するために CPU 実行に切り替える
        """
        assert self.model is not None
        original_device = self.device

        try:
            if CPU_EVALUATION_ON_MPS_BACKEND and original_device.type == "mps":
                self.device = torch.device("cpu")
                self.model = self.model.to(self.device)

            yield
        finally:
            # 元のデバイスに戻す
            self.device = original_device
            self.model.to(original_device)


def detect_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def flatten_nested_dict(
    d: dict[str, Any], parent_key: str = "", sep: str = "."
) -> dict[str, Any]:
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_nested_dict(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items


# NOTE: This is based on the Hugging Face implementation
# https://github.com/huggingface/transformers/blob/5f4ecf2d9f867a1255131d2461d75793c0cf1db2/src/transformers/optimization.py#L142C1-L173C54
def get_cosine_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        num_cycles (`float`, *optional*, defaults to 0.5):
            The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
            following a half-cosine).
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    lr_lambda = partial(
        _get_cosine_schedule_with_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=num_cycles,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)


def _get_cosine_schedule_with_warmup_lr_lambda(
    current_step: int,
    *,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float,
):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    progress = float(current_step - num_warmup_steps) / float(
        max(1, num_training_steps - num_warmup_steps)
    )
    return max(
        0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
    )
