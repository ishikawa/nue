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
import os
from typing import Any

import click
import mlx.core as mx
import mlx.data
import mlx.nn as nn
import numpy as np
from datasets import Dataset
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

        click.secho("[3/7] Prepare dataset", fg="green", bold=True)
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

        with yaspin().cyan as spinner:
            for i, input_ids in enumerate(train_stream):
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

                # print(f"{i} batch: {input_ids}")
                # print(f"{i} attn_mask: {attn_mask}")
                # print(f"{i} labels: {labels}")

                mx.eval(input_ids)
                mx.eval(attn_mask)
                mx.eval(labels)

                logits = self.model(input_ids, attention_mask=attn_mask)
                print(f"{i} logits: {logits}")

                if i >= 5:
                    break
