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
import time
from contextlib import contextmanager
from typing import Iterable, Optional, cast

import click
import torch
from datasets import Dataset
from termcolor import colored

# from torch.amp.grad_scaler import GradScaler
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchtune.training import get_cosine_schedule_with_warmup
from yaspin import yaspin
from yaspin.core import Yaspin

from nue.model.torch import MinimalGPT, init_weights
from nue.train.trainer import BaseTrainer

from .base import TrainingOptions
from .tokenizer import IGNORE_TOKEN_ID, PAD_TOKEN_ID, TOKENIZER

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

    def _train(
        self,
        *,
        log_validation_max_tokens: int,
        measure_time: bool,
        override_base_lr: float | None,
    ) -> None:
        options = self.options
        model = self.model

        assert model is not None
        assert self.train_dataset is not None
        assert self.validation_dataset is not None
        assert self.train_loader is not None
        assert self.validation_loader is not None

        # --------- 4) Optimizer & Scheduler ---------
        click.secho("[4/7] Prepare optimizer & scheduler", fg="green", bold=True)

        # NOTE: Adam だと速く収束するが鋭い谷に落ちやすい
        # optimizer = optim.Adam(model.parameters(), lr=lr)
        # NOTE: SGD は安定性を増すが、データ量が少ない時は収束しなかった
        optimizer = AdamW(
            model.parameters(),
            lr=options.lr,
            # 過学習防止のため正則化
            weight_decay=0.01,
            fused=True,
        )

        # --- スケジューラーの設定 ---
        # おおよその総学習ステップ数を計算 (エポック数 x 1エポックあたりのステップ数)
        # len(dataset) はチャンク化後の訓練データセットのサンプル数

        # 学習率スケジューラー: 線形ウォームアップ後、トレーニング終了まで余弦減衰
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.num_warmup_steps,
            num_training_steps=self.num_training_steps,
        )

        criterion = torch.nn.CrossEntropyLoss(
            reduction="mean",
            # Label smoothing for better generalization
            label_smoothing=0.1,
            # Padding token ID
            ignore_index=IGNORE_TOKEN_ID,
        )

        # --------- 5) 前回の学習状態を復元 ---------
        parameters_path = os.path.join(options.model_dir, "parameters.pt")
        checkpoint_path = os.path.join(options.model_dir, "checkpoint.pt")

        checkpoint = None
        start_epoch = 0
        start_step = 0

        if os.path.exists(checkpoint_path):
            click.secho(
                "[5/7] Resuming training from checkpoint", fg="green", bold=True
            )
            click.secho(f"Loading checkpoint from {checkpoint_path}", fg="cyan")

            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            model.load_state_dict(checkpoint["model_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])

            last_epoch = int(checkpoint["epoch"])
            last_step = int(checkpoint["step"])

            start_epoch = last_epoch + 1
            start_step = last_step + 1

            scheduler.last_epoch = last_step

            click.secho(f"Resuming training from epoch {start_epoch + 1}", fg="black")
        else:
            click.secho("[5/7] Training from scratch", fg="bright_green", bold=True)
            os.makedirs(options.model_dir, exist_ok=True)

        # --------- 6) 学習ループ ---------
        def save_checkpoint(
            spinner: Yaspin, i_epoch: int, i_step: int, *, verbose: bool = False
        ):
            if verbose:
                spinner.write(
                    colored(
                        f"Saving checkpoint to {checkpoint_path}",
                        "cyan",
                    )
                )
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "epoch": i_epoch,
                    "step": i_step,
                },
                checkpoint_path,
            )

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

        click.secho("[6/7] Start training loop", fg="green", bold=True)
        model.train()

        loss = None
        logits_mean = None

        # MPS では AMP (自動混合精度) を使用しない
        use_amp = self.device.type != "mps"

        # bfloat16 は十分な精度を持つので、GradScaler 不要
        # grad_scaler = GradScaler(self.device.type, enabled=use_amp)
        with yaspin().cyan as spinner:

            def set_spinner_text(
                i_epoch: int,
                i_step: int,
                *,
                lr: float,
                loss: Optional[float] = None,
                logits_mean: Optional[float] = None,
            ):
                p = (i_step + 1) / self.num_training_steps_per_epoch
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

            for i_epoch in range(start_epoch, options.n_epochs):
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
                loader_iter = iter(self.train_loader)

                while True:
                    try:
                        # 学習再開時には、開始前のデータをスキップする
                        if i_epoch == start_epoch and i_step < start_step:
                            next(loader_iter)
                            continue

                        set_spinner_text(
                            i_epoch=i_epoch,
                            i_step=i_step,
                            lr=scheduler.get_last_lr()[0],
                            loss=loss,
                            logits_mean=logits_mean,
                        )

                        if measure_time:
                            torch.mps.synchronize()
                            t0 = time.perf_counter()

                        batch = next(loader_iter)

                        # Runs the forward pass under `autocast`.
                        with torch.autocast(
                            device_type=self.device.type,
                            dtype=torch.bfloat16,
                            enabled=use_amp,
                        ):
                            if measure_time:
                                torch.mps.synchronize()
                                t1 = time.perf_counter()

                            input_ids = batch["input_ids"].to(self.device)
                            attention_mask = batch["attention_mask"].to(self.device)
                            labels = batch["labels"].to(self.device)

                            logits = model(
                                input_ids,
                                attention_mask=attention_mask,
                            )

                            if measure_time:
                                torch.mps.synchronize()
                                t2 = time.perf_counter()

                            loss = criterion(
                                logits.view(-1, self.config.vocab_size),
                                labels.view(-1),
                            )

                        # Exits `autocast` before backward().
                        logits_mean = logits.abs().mean().item()

                        set_spinner_text(
                            i_epoch=i_epoch,
                            i_step=i_step,
                            lr=scheduler.get_last_lr()[0],
                            loss=loss.item(),
                            logits_mean=logits_mean,
                        )

                        # 勾配の計算
                        # grad_scaler.scale(loss).backward()
                        loss.backward()

                        if measure_time:
                            torch.mps.synchronize()
                            t3 = time.perf_counter()

                        # 勾配のクリッピング
                        # ただし、小規模モデルは毎 step でなくても安定する
                        if i_step % 4 == 0:
                            # Un-scales the gradients of optimizer's assigned parameters in-place
                            # grad_scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                        # パラメータの更新
                        # grad_scaler.step(optimizer)
                        # grad_scaler.update()
                        optimizer.step()
                        scheduler.step()

                        # 勾配の初期化
                        optimizer.zero_grad()

                        if measure_time:
                            torch.mps.synchronize()
                            t4 = time.perf_counter()

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
                                model.eval()

                                # Evaluate on validation dataset
                                val_loss = self.evaluate(
                                    self.validation_loader,
                                    criterion,
                                    max_tokens=log_validation_max_tokens,
                                )

                                # Generate samples
                                with torch.no_grad():
                                    for text in self.generate_samples():
                                        spinner.write(
                                            colored(
                                                f"  SAMPLE: {text}",
                                                "yellow",
                                            )
                                        )
                            finally:
                                model.train()

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

                            lr = scheduler.get_last_lr()[0]
                            progress += f"{colored('lr=', 'cyan')}{lr:.6f} "

                            if measure_time:
                                progress += "("
                                progress += (
                                    f"{step_elapsed / options.log_interval:.3f}s "
                                )
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

                        # Save model checkpoint
                        if (i_step + 1) % options.save_interval == 0:
                            # Epoch is not finished yet, so -1
                            save_checkpoint(spinner, i_epoch - 1, i_step)

                    except StopIteration:
                        break
                    finally:
                        i_step += 1

                # エポック終わりのサンプル生成と評価
                try:
                    model.eval()

                    # 評価
                    val_loss = self.evaluate(self.validation_loader, criterion)

                    # サンプル生成
                    with torch.no_grad():
                        for text in self.generate_samples():
                            spinner.write(
                                colored(
                                    f"  Sample generation: {text}",
                                    "yellow",
                                )
                            )
                finally:
                    model.train()

                avg_epoch_loss = epoch_loss / len(self.train_dataset)

                spinner.write(
                    colored(
                        f"Epoch {i_epoch + 1}/{options.n_epochs} finished",
                        "magenta",
                        attrs=["bold"],
                    )
                    + colored(
                        f" (avg loss={avg_epoch_loss:.4f}, val loss={val_loss:.4f})",
                        "magenta",
                    )
                )

                # Save epoch checkpoint
                save_checkpoint(spinner, i_epoch, 0, verbose=True)

        # --------- モデル保存 ---------
        click.secho(
            f"[7/7] Saving model to {parameters_path}...", fg="bright_green", bold=True
        )

        torch.save(model.state_dict(), parameters_path)

    def _generate_text(
        self,
        prompt: str,
        *,
        max_new_length: int,
    ) -> str:
        assert self.model is not None

        ids: list[int] = TOKENIZER.EncodeAsIds(prompt)
        idx = torch.tensor([ids], dtype=torch.long).to(self.device)
        out = self._generate(idx, max_new_tokens=max_new_length)[0].cpu().tolist()

        return TOKENIZER.DecodeIds(out)

    @torch.no_grad()
    def _generate(self, idx: torch.Tensor, max_new_tokens: int = 32):
        """Greedy text generation (for demo)."""
        assert self.model is not None
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.ctx_len :]
            logits = self.model(idx_cond)[:, -1, :]  # [B,vocab]
            next_tok = logits.argmax(dim=-1, keepdim=True)
            idx = torch.cat([idx, next_tok], dim=1)
        return idx

    def generate_samples(self) -> Iterable[str]:
        with torch.no_grad():
            with self.use_cpu_on_mps():
                for prompt in ["富士山は", "Alan Turing is "]:
                    yield self._generate_text(prompt, max_new_length=50)

    def evaluate(
        self,
        dataloader: DataLoader,
        criterion: torch.nn.Module,
        *,
        max_tokens: Optional[int] = None,
    ) -> float:
        assert self.model is not None

        total_loss = 0.0
        total_tokens = 0

        with torch.no_grad():
            with self.use_cpu_on_mps():
                for batch in dataloader:
                    input_ids = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)
                    labels = batch["labels"].to(self.device)

                    logits = self.model(
                        input_ids,
                        attention_mask=attention_mask,
                    )

                    loss = criterion(
                        logits.view(-1, self.config.vocab_size),
                        labels.view(-1),
                    )

                    num_tokens = (labels != IGNORE_TOKEN_ID).sum().item()
                    total_loss += loss.item() * num_tokens
                    total_tokens += num_tokens

                    if max_tokens is not None and total_tokens >= max_tokens:
                        break

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
