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

import click
import mlx.core as mx
import mlx.nn as nn

from nue.mlx.model import NueLM
from nue.model.base import GPTConfig
from nue.train.base import TrainingOptions, TrainingSession
from nue.train.dataset import load_train_and_validation_dataset, load_train_dataset
from nue.train.tokenizer import TOKENIZER


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
        click.secho("[3/7] Prepare dataset", fg="green", bold=True)
        dataset_result = load_train_and_validation_dataset(
            ctx_len=options.ctx_len,
            chunk_overlap_len=options.chunk_overlap_len,
            override_data_size=options.override_data_size,
        )

        train_dataset = dataset_result.train_dataset
        validation_dataset = dataset_result.validation_dataset
        total_tokens = dataset_result.total_tokens
