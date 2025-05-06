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
import re
import time
from dataclasses import dataclass
from typing import Iterable, Optional, cast

import click
import torch
from datasets import (
    Dataset,
    Features,
    Sequence,
    Value,
    concatenate_datasets,
    load_dataset,
)
from sentencepiece import SentencePieceProcessor
from termcolor import colored
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from yaspin import yaspin
from yaspin.core import Yaspin

from nue.common import BUILD_DIR, DATASET_CACHE_DIR
from nue.datasets import DATASET_LIST
from nue.gpt import GPTConfig, MinimalGPT, init_weights

# グローバルにロードしておくと子プロセスが継承できる
TOKENIZER = SentencePieceProcessor()
TOKENIZER.Load(str(BUILD_DIR / "tokenizer.model"))

PAD_ID: int = TOKENIZER.pad_id()  # 例: 0
IGNORE = -100  # CrossEntropyLoss が無視する値

assert PAD_ID is not None
assert isinstance(PAD_ID, int)
assert PAD_ID >= 0, "PAD_ID must be non-negative"

# NOTE: torch >= 2.6.0 かつ MPS Backend だと SDPA で NaN が出ることがある
#
# [MPS] MultiheadAttention with masks and dropout produces NaNs #151667
# https://github.com/pytorch/pytorch/issues/151667
#
# このフラグを True にすると、評価時は CPU で推論する
CPU_EVALUATION_ON_MPS_BACKEND = False


@dataclass(frozen=True)
class Epoch:
    epoch: int
    loss: float


@dataclass(frozen=True)
class TrainingOptions:
    n_epochs: int
    batch_size: int
    ctx_len: int
    # 学習データセットのチャンク間のオーバーラップ長
    chunk_overlap_len: int
    n_embed: int
    n_heads: int
    n_layers: int
    mlp_ratio: int
    seed: int
    lr: float
    lr_scheduler_patience: int
    log_interval: int
    save_interval: int
    model_dir: str


@dataclass
class TrainingSession:
    epochs: list[Epoch]
    options: TrainingOptions


def tokenize_and_chunk(
    text: str, tokenizer: SentencePieceProcessor, ctx_len: int, overlap_len: int
) -> list[list[int]]:
    """テキストをトークナイズし、オーバーラップ付きのチャンクに分割する"""
    tokens = tokenizer.EncodeAsIds(text)

    if not tokens:
        return []

    # トークン数がコンテキスト長より短い場合は、そのまま単一のチャンクとして返す
    if len(tokens) <= ctx_len:
        return [tokens]

    chunks = []
    stride = ctx_len - overlap_len

    if stride <= 0:
        raise ValueError(
            f"overlap_len ({overlap_len}) must be smaller than ctx_len ({ctx_len})"
        )

    for i in range(0, len(tokens), stride):
        chunk = tokens[i : i + ctx_len]
        if not chunk:  # まれにループの最後で空になる場合
            continue

        # チャンクを追加（最後のチャンクがctx_len未満でもOK）
        chunks.append(chunk)

        # 次のチャンクの開始位置が元のトークン長を超える場合、ループを終了
        if i + stride >= len(tokens):
            break

    # 念の為、チャンクが生成されなかった場合のフォールバック
    if not chunks and tokens:
        chunks.append(tokens[:ctx_len])

    return chunks


# --- 第1段階の map で使用する関数 ---
def map_tokenize_to_chunk_lists(
    examples: dict[str, list], column: str, ctx_len: int, overlap_len: int
) -> dict[str, list]:
    """
    各テキストをチャンク化し、チャンクのリストと、各チャンクの長さのリストを返す。
    出力の行数は入力と同じ。
    """
    output = {"input_ids_chunks": [], "num_tokens_chunks": []}
    # TOKENIZER はグローバル変数から参照
    for text in examples[column]:
        chunks = tokenize_and_chunk(text, TOKENIZER, ctx_len, overlap_len)
        output["input_ids_chunks"].append(chunks)
        output["num_tokens_chunks"].append([len(c) for c in chunks])
    return output


def collate_pad(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    # 1) 各シーケンス
    seqs = [b["input_ids"] for b in batch]

    # 2) 右パディング
    padded = pad_sequence(seqs, batch_first=True, padding_value=PAD_ID)  # [B, T]

    # 3) attention_mask （1=実トークン、0=PAD）
    attn_mask = (padded != PAD_ID).long()  # [B, T]

    # 4) Labels の作成
    # - 次トークン予測タスクでは、labels[i] が input_ids[i+1] に対応
    # - パディング部分は損失計算から除外する必要がある

    # 全ての要素を IGNORE (損失計算で無視される値) で初期化
    labels = torch.full_like(padded, IGNORE)

    # input_ids を左シフトして labels に代入することで、
    # labels[i] <- input_ids[i+1]
    labels[:, :-1] = padded[:, 1:]

    # ラベルが PAD_ID となっている箇所を IGNORE にする
    labels[labels == PAD_ID] = IGNORE

    return {
        "input_ids": padded,
        "attention_mask": attn_mask,
        "labels": labels,
    }


class PyTorchTrainer:
    config: GPTConfig
    options: TrainingOptions
    model: MinimalGPT | None = None

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

    def load_dataset(self, *, override_data_size: Optional[str] = None) -> Dataset:
        datasets_list: list[Dataset] = []

        for dataset_config in DATASET_LIST:
            split = dataset_config.train_split

            if override_data_size is not None:
                if m := re.match(r"^(\w+)", split):
                    split = f"{m.group(1)}[:{override_data_size}]"
                else:
                    split = f"{split}[:{override_data_size}]"

            dataset = load_dataset(
                dataset_config.path,
                dataset_config.name,
                split=split,
                cache_dir=str(DATASET_CACHE_DIR),
                trust_remote_code=dataset_config.trust_remote_code,
            )

            # Tokenize
            mapped_dataset = dataset.map(
                map_tokenize_to_chunk_lists,
                fn_kwargs={
                    "column": dataset_config.content_column,
                    "ctx_len": self.config.ctx_len,
                    "overlap_len": self.options.chunk_overlap_len,
                },
                # 元のカラムを削除
                remove_columns=dataset.column_names,  # type: ignore
                batched=True,
                num_proc=os.cpu_count(),  # type: ignore
                desc=f"Tokenizing and chunking dataset for '{dataset_config.name}'",  # type: ignore
            )

            # チャンクリストが空の行を除外 (元のテキストが空など)
            filtered_dataset = mapped_dataset.filter(
                lambda example: len(example["input_ids_chunks"]) > 0,
                num_proc=os.cpu_count(),
                desc=f"Filtering empty rows for '{dataset_config.name}'",
            )

            # --- 第2段階: フラット化 ---
            # フラット化後のデータセットのスキーマ (特徴量) を定義
            new_features = Features(
                {
                    "input_ids": Sequence(feature=Value(dtype="int32")),
                    "num_tokens": Value(dtype="int32"),
                }
            )

            def flatten_batch(examples: dict[str, list]) -> dict[str, list]:
                flat_input_ids = []
                flat_num_tokens = []

                for i in range(len(examples["input_ids_chunks"])):
                    for j in range(len(examples["input_ids_chunks"][i])):
                        flat_input_ids.append(examples["input_ids_chunks"][i][j])
                        flat_num_tokens.append(examples["num_tokens_chunks"][i][j])

                return {"input_ids": flat_input_ids, "num_tokens": flat_num_tokens}

            # リストをフラット化する
            flat_dataset = filtered_dataset.map(
                flatten_batch,
                batched=True,
                # フラット化前のネストしたカラムを削除
                remove_columns=["input_ids_chunks", "num_tokens_chunks"],
                # 新しいスキーマを指定
                features=new_features,
                num_proc=os.cpu_count(),
                desc=f"Flattening chunks for '{dataset_config.name}' (parallel)",
            )
            print(
                colored(
                    f"Successfully flattened '{dataset_config.name}' using map(batched=True, num_proc={os.cpu_count()})",
                    "green",
                )
            )

            processed_dataset = cast(Dataset, flat_dataset)
            datasets_list.append(processed_dataset)

        final_dataset = concatenate_datasets(datasets_list)

        # 合計トークン数
        if "num_tokens" not in final_dataset.column_names:
            print(
                colored(
                    "Warning: 'num_tokens' column not found after flattening. Re-calculating.",
                    "yellow",
                )
            )
            final_dataset = final_dataset.map(
                lambda x: {"num_tokens": len(x["input_ids"])}, num_proc=os.cpu_count()
            )
        self.total_tokens = sum(final_dataset["num_tokens"])

        final_dataset.set_format(type="torch", columns=["input_ids"])
        return final_dataset

    def train(
        self,
        session: TrainingSession,
        *,
        override_data_size: Optional[str] = None,
        measure_time: bool = False,
    ) -> None:
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

        # Save hyperparameters in JSON format
        with open(os.path.join(session.options.model_dir, "hparams.json"), "w") as f:
            json.dump(dataclasses.asdict(self.config), f, indent=4)

        # --------- 2) Minimal GPT 初期化 ---------
        click.secho("[2/7] Initialize Minimal GPT", fg="green", bold=True)

        model = MinimalGPT(self.config).to(torch.bfloat16).to(device)
        model.apply(init_weights)

        self.model = model

        # --------- 3) データセット準備 ---------
        click.secho("[3/7] Prepare dataset", fg="green", bold=True)
        dataset = self.load_dataset(override_data_size=override_data_size)

        # 合計トークン数を計算
        total_tokens = sum(dataset["num_tokens"])

        # train/test split
        train_and_test_datasets = dataset.train_test_split(test_size=0.05)
        validation_dataset = train_and_test_datasets["test"]
        dataset = train_and_test_datasets["train"]

        # DataLoader
        loader = DataLoader(
            dataset,  # type: ignore
            batch_size=options.batch_size,
            shuffle=True,
            collate_fn=collate_pad,
        )
        validation_loader = DataLoader(
            validation_dataset,  # type: ignore
            batch_size=options.batch_size,
            shuffle=False,
            collate_fn=collate_pad,
        )

        click.secho(
            f"Total tokens: {format_number_abbrev(total_tokens)} ({total_tokens:,})",
            fg="cyan",
        )
        click.secho(
            f"Loader created (train: {len(dataset):,} rows, val: {len(validation_dataset):,} rows)",
            fg="cyan",
        )

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
        )

        # --- スケジューラーの設定 ---
        # おおよその総学習ステップ数を計算 (エポック数 x 1エポックあたりのステップ数)
        # len(dataset) はチャンク化後の訓練データセットのサンプル数
        num_training_steps_per_epoch = math.ceil(len(dataset) / options.batch_size)
        num_training_steps = num_training_steps_per_epoch * options.n_epochs

        # ウォームアップステップ数: 1エポックあたりのステップ数の5%または5000
        num_warmup_steps = int(min(num_training_steps_per_epoch * 0.05, 5000))

        click.secho(
            f"Estimated total steps: {num_training_steps}, Warmup steps: {num_warmup_steps}",
            fg="cyan",
        )

        # 学習率スケジューラー: 線形ウォームアップ後、余りを余弦減衰
        def lr_lambda(
            current_step: int, num_warmup_steps: int, num_training_steps: int
        ):
            # current_step が総学習ステップ数を超えたら、学習率は0
            if current_step >= num_training_steps:
                return 0.0

            # 線形ウォームアップ
            if num_warmup_steps > 0 and current_step < num_warmup_steps:
                return float(current_step) / float(num_warmup_steps)

            # ウォームアップがないか、ウォームアップ期間終了後のコサイン減衰
            # num_training_steps と num_warmup_steps が同じ場合（つまりウォームアップのみで減衰なし、またはステップ数が不足）を考慮
            decay_steps = num_training_steps - num_warmup_steps
            if (
                decay_steps <= 0
            ):  # 減衰期間がない、または計算がおかしい場合は、ウォームアップ後の最大学習率を維持するか、0にする
                return 1.0  # ここは状況によるが、ウォームアップのみの場合は1.0を維持、あるいはエラーを出した方が良い場合も
                # もし num_warmup_steps == num_training_steps なら、最後のステップなので 0 に近い値を返すのが適切か

            progress = float(current_step - num_warmup_steps) / float(decay_steps)
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        # スケジューラーの初期化時に num_warmup_steps と num_training_steps を渡すように変更
        # scheduler = LambdaLR(optimizer, lr_lambda) # 元の呼び出し方
        scheduler = LambdaLR(
            optimizer,
            lambda step: lr_lambda(step, num_warmup_steps, num_training_steps),
        )

        criterion = torch.nn.CrossEntropyLoss(
            reduction="mean",
            # Label smoothing for better generalization
            label_smoothing=0.1,
            # Padding token ID
            ignore_index=IGNORE,
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

                        batch = next(loader_iter)

                        if measure_time:
                            torch.mps.synchronize()
                            t1 = time.perf_counter()

                        input_ids = batch["input_ids"].to(device)
                        attention_mask = batch["attention_mask"].to(device)
                        labels = batch["labels"].to(device)

                        logits = self.model(
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

                        # 勾配の計算
                        loss.backward()

                        if measure_time:
                            torch.mps.synchronize()
                            t3 = time.perf_counter()

                        # 勾配のクリッピング
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                        # パラメータの更新
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

                        # Log training progress
                        if (i_step + 1) % options.log_interval == 0:
                            # Evaluate on validation dataset
                            # 最低限安定して計測できるように 50,000 トークンまで
                            try:
                                model.eval()

                                # 評価
                                val_loss = self.evaluate(
                                    validation_loader, criterion, max_tokens=50_000
                                )

                                # サンプル生成
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

                # エポック終わりのサンプル生成と評価
                try:
                    model.eval()

                    # 評価
                    val_loss = self.evaluate(validation_loader, criterion)

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

        ids: list[int] = TOKENIZER.EncodeAsIds(prompt)
        idx = torch.tensor([ids], dtype=torch.long).to(device)
        out = self.model._generate(idx, max_new_tokens=max_new_length)[0].cpu().tolist()

        return TOKENIZER.DecodeIds(out)

    def generate_samples(self) -> Iterable[str]:
        for prompt in ["富士山は", "東京の", "Alan Turing is "]:
            yield self.generate_text(prompt, max_new_length=50)

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

        device = detect_device()
        original_device = device

        with torch.no_grad():
            try:
                # NOTE: MPS BUG? MPS Backend だと推論時に NaN が出ることがあるので、CPU で推論する
                if CPU_EVALUATION_ON_MPS_BACKEND and device.type == "mps":
                    device = torch.device("cpu")
                    self.model = self.model.to(device)

                for batch in dataloader:
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels = batch["labels"].to(device)

                    logits = self.model(
                        input_ids,
                        attention_mask=attention_mask,
                    )

                    loss = criterion(
                        logits.view(-1, self.model.cfg.vocab_size),
                        labels.view(-1),
                    )

                    num_tokens = (labels != IGNORE).sum().item()
                    total_loss += loss.item() * num_tokens
                    total_tokens += num_tokens

                    if max_tokens is not None and total_tokens >= max_tokens:
                        break
            finally:
                # 元のデバイスに戻す
                self.model.to(original_device)
                pass

        avg_loss = total_loss / total_tokens
        return avg_loss


def detect_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def format_number_abbrev(n: int) -> str:
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.1f}B"
    elif n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    elif n >= 1_000:
        return f"{n / 1_000:.1f}K"
    else:
        return str(n)
