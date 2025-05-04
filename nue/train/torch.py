import dataclasses
import json
import os
import time
from typing import Optional

import click
import torch
from termcolor import colored
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from yaspin import yaspin
from yaspin.core import Yaspin

from nue.minigpt import MinimalGPT, init_weights

from .base import BaseTrainer
from .models import Epoch, TrainingSession


class PyTorchTrainer(BaseTrainer):
    model: MinimalGPT | None = None

    def name(self) -> str:
        return "pytorch"

    def evaluate(
        self,
        dataloader: DataLoader,
        criterion: torch.nn.Module,
        *,
        max_tokens: Optional[int] = None,
    ) -> float:
        device = detect_device()

        assert self.model is not None

        total_loss = 0.0
        total_tokens = 0

        with torch.no_grad():
            for batch in dataloader:
                # batch["ids"] が (B, L) の tensor だとして
                inputs = batch["ids"][:, :-1].to(device)  # ひとトークンずらし
                labels = batch["ids"][:, 1:].to(device)
                outputs = self.model(inputs)  # 出力は (B, L, vocab_size)
                logits = outputs.logits if hasattr(outputs, "logits") else outputs
                loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
                num_tokens = (labels != -100).sum().item()
                total_loss += loss.item() * num_tokens
                total_tokens += num_tokens
                if max_tokens is not None and total_tokens >= max_tokens:
                    break

        avg_loss = total_loss / total_tokens
        return avg_loss

    def _train(self, session: TrainingSession, *, measure_time: bool = False) -> None:
        options = session.options

        # シード設定
        if options.seed is not None:
            torch.manual_seed(options.seed)

        # --------- 1) デバイス設定 (MPS) ---------
        click.secho("[1/7] Device setup", fg="green", bold=True)

        device = detect_device()

        click.secho(
            f"vocab_size: {self.config.vocab_size}, device: {device}", fg="cyan"
        )

        # --------- 2) Minimal GPT 初期化 ---------
        click.secho("[2/7] Initialize Minimal GPT", fg="green", bold=True)

        model = MinimalGPT(self.config).to(torch.bfloat16).to(device)
        model.apply(init_weights)

        self.model = model

        # --------- 3) データセット準備 ---------
        click.secho("[3/7] Prepare dataset", fg="green", bold=True)
        dataset = self.load_dataset()
        dataset.set_format(type="torch", columns=["ids"])

        # train/test split
        train_and_test_datasets = dataset.train_test_split(test_size=0.05)
        validation_dataset = train_and_test_datasets["test"]
        dataset = train_and_test_datasets["train"]

        # DataLoader
        def collate(
            batch: list[dict[str, torch.Tensor]],
        ) -> tuple[torch.Tensor, torch.Tensor]:
            # batch: list of dicts with "ids"
            data = torch.stack([ex["ids"] for ex in batch], dim=0)
            x = data[:, :-1].to(device)  # [B, T]
            y = data[:, 1:].to(device)  # [B, T]
            return x, y

        # DataLoaderを作成
        loader = DataLoader(
            dataset,  # type: ignore
            batch_size=options.batch_size,
            shuffle=True,
            collate_fn=collate,
        )
        validation_loader = DataLoader(
            validation_dataset,  # type: ignore
            batch_size=options.batch_size,
            shuffle=False,
        )

        click.secho(
            f"Loader created (train: {len(dataset)}, validation: {len(validation_dataset)})",
            fg="cyan",
        )

        # --------- 4) Optimizer & Scheduler ---------
        click.secho("[4/7] Prepare optimizer & scheduler", fg="green", bold=True)
        # NOTE: Adam だと速く収束するが鋭い谷に落ちやすい
        # optimizer = optim.Adam(model.parameters(), lr=lr)
        # NOTE: SGD は安定性を増すが、データ量が少ない時は収束しなかった
        optimizer = AdamW(
            model.parameters(),
            lr=3e-4,
            # 過学習防止のため正則化
            weight_decay=0.01,
        )

        # Define learning rate scheduler
        # Reduce learning rate when loss plateaus (patience=5)
        # NOTE: 一般的にはコザイン減衰のスケジューラーが使われているらしい
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            min_lr=1e-4,
            patience=options.lr_scheduler_patience,
            threshold=1e-3,  # relative 0.1 % 下降を許容
            threshold_mode="rel",
        )

        pad_token_id: int | None = self.tokenizer.pad_id()
        criterion = torch.nn.CrossEntropyLoss(
            reduction="mean",
            # Label smoothing for better generalization
            label_smoothing=0.1,
            # Padding token ID
            ignore_index=-100
            if pad_token_id is None or pad_token_id < 0
            else pad_token_id,
        )

        # --------- 5) 前回の学習状態を復元 ---------
        parameters_path = os.path.join(options.model_dir, "parameters.pt")
        checkpoint_path = os.path.join(options.model_dir, "checkpoint.pt")
        session_path = os.path.join(options.model_dir, "train.json")

        checkpoint = None
        start_epoch = 0
        start_step = 0

        if os.path.exists(checkpoint_path):
            click.secho(
                "[5/7] Resuming training from checkpoint", fg="green", bold=True
            )
            click.secho(f"Loading checkpoint from {checkpoint_path}", fg="cyan")

            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint["model_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])

            start_epoch = int(checkpoint["epoch"]) + 1
            start_step = int(checkpoint["step"]) + 1

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
            with open(session_path, "w") as f:
                json.dump(
                    dataclasses.asdict(session),
                    f,
                    indent=4,
                )

        click.secho("[6/7] Start training loop", fg="green", bold=True)
        model.train()

        loss = None
        logits_mean = None

        with yaspin().cyan as spinner:

            def set_spinner_text(
                i_epoch: int,
                i_step: int,
                *,
                lr: float,
                loss: Optional[float] = None,
                logits_mean: Optional[float] = None,
            ):
                spinner.text = (
                    colored(
                        f"Epoch {i_epoch + 1}/{options.n_epochs} Step {i_step + 1}",
                        color="cyan",
                    )
                    + " ("
                    + f"lr: {lr:.4f}"
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

                i_step = 0
                loader_iter = iter(loader)

                while True:
                    try:
                        if i_step < start_step:
                            next(loader_iter)
                            continue

                        set_spinner_text(
                            i_epoch=i_epoch,
                            i_step=i_step,
                            lr=scheduler.get_last_lr()[0],
                            loss=loss,
                            logits_mean=logits_mean,
                        )

                        # ここで bfloat16 モードに切り替え
                        # NOTE: MPS ではほとんどパフォーマンスの違いがないためコメントアウト
                        # with torch.autocast(device_type="mps", dtype=torch.bfloat16):
                        if measure_time:
                            torch.mps.synchronize()
                            t0 = time.perf_counter()

                        x, y = next(loader_iter)

                        if measure_time:
                            torch.mps.synchronize()
                            t1 = time.perf_counter()

                        logits = self.model(x)  # [B, T, vocab]

                        if measure_time:
                            torch.mps.synchronize()
                            t2 = time.perf_counter()

                        loss = criterion(
                            logits.view(-1, self.config.vocab_size),
                            y.view(-1),
                        )

                        logits_mean = logits.abs().mean().item()
                        set_spinner_text(
                            i_epoch=i_epoch,
                            i_step=i_step,
                            lr=scheduler.get_last_lr()[0],
                            loss=loss.item(),
                            logits_mean=logits_mean,
                        )

                        # MPS では GradScaler 要らず。そのまま backward → step
                        # NOTE: MPS ではほとんどパフォーマンスの違いがないためコメントアウト

                        # 1. 勾配の初期化
                        optimizer.zero_grad()
                        # 2. 勾配の計算
                        loss.backward()

                        if measure_time:
                            torch.mps.synchronize()
                            t3 = time.perf_counter()

                        # 3. 勾配のクリッピング
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        # 4. パラメータの更新
                        optimizer.step()

                        if measure_time:
                            torch.mps.synchronize()
                            t4 = time.perf_counter()

                        total_loss += loss.item()
                        epoch_loss += loss.item()

                        io_elapsed += t1 - t0
                        forward_elapsed += t2 - t1
                        backward_elapsed += t3 - t2
                        optimizer_elapsed += t4 - t3

                        # Log training progress
                        if (i_step + 1) % options.log_interval == 0:
                            # Evaluate on validation dataset
                            # 最低限安定して計測できるように 50,000 トークンまで
                            try:
                                model.eval()
                                val_loss = self.evaluate(
                                    validation_loader, criterion, max_tokens=50_000
                                )
                            finally:
                                model.train()

                            scheduler.step(val_loss)

                            progress = (
                                f"  Step {i_step + 1} "
                                + f"{colored('loss=', 'cyan')}{total_loss / options.log_interval:.3f} "
                                + f"{colored('val_loss=', 'cyan')}{val_loss:.3f} "
                            )

                            lr = scheduler.get_last_lr()[0]
                            progress += f"{colored('lr=', 'cyan')}{lr:.6f} "

                            if measure_time:
                                progress += f"{colored('io=', 'cyan')}{io_elapsed / options.log_interval:.3f}s "
                                progress += f"{colored('forward=', 'cyan')}{forward_elapsed / options.log_interval:.3f}s "
                                progress += f"{colored('backward=', 'cyan')}{backward_elapsed / options.log_interval:.3f}s "
                                progress += f"{colored('optimizer=', 'cyan')}{optimizer_elapsed / options.log_interval:.3f}s "

                            spinner.write(progress)

                            total_loss = 0.0
                            io_elapsed = 0.0
                            forward_elapsed = 0.0
                            backward_elapsed = 0.0
                            optimizer_elapsed = 0.0

                        # Save model checkpoint
                        if (i_step + 1) % options.save_interval == 0:
                            # Epoch is not finished yet, so -1
                            save_checkpoint(spinner, i_epoch - 1, i_step)

                    except StopIteration:
                        break
                    finally:
                        i_step += 1

                # エポック終わりのサンプル生成
                try:
                    model.eval()

                    with torch.no_grad():
                        prompt = "昔々"
                        ids: list[int] = self.tokenizer.EncodeAsIds(prompt)
                        idx = torch.tensor([ids], dtype=torch.long).to(device)
                        out = model._generate(idx, max_new_tokens=50)[0].cpu().tolist()

                        spinner.write(
                            colored(
                                f"  Sample generation: {self.tokenizer.DecodeIds(out)}",
                                "yellow",
                            )
                        )

                    val_loss = self.evaluate(validation_loader, criterion)
                    scheduler.step(val_loss)
                finally:
                    model.train()

                avg_epoch_loss = epoch_loss / len(loader)
                session.epochs.append(Epoch(epoch=i_epoch + 1, loss=avg_epoch_loss))

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

        with open(session_path, "w") as f:
            json.dump(
                dataclasses.asdict(session),
                f,
                indent=4,
            )

        torch.save(model.state_dict(), parameters_path)

    def generate_text(
        self,
        prompt: str,
        *,
        max_new_length: int,
    ) -> str:
        assert self.model is not None

        device = detect_device()

        ids: list[int] = self.tokenizer.EncodeAsIds(prompt)
        idx = torch.tensor([ids], dtype=torch.long).to(device)
        out = self.model._generate(idx, max_new_tokens=max_new_length)[0].cpu().tolist()

        return self.tokenizer.DecodeIds(out)


def detect_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")
