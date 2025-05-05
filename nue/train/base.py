import dataclasses
import json
import os
from abc import ABC, abstractmethod
from typing import Any, cast

import click
from datasets import Dataset, concatenate_datasets, load_dataset
from sentencepiece import SentencePieceProcessor

from nue.common import BUILD_DIR, DATASET_CACHE_DIR
from nue.datasets import DATASET_LIST
from nue.minigpt import GPTConfig

from .models import TrainingOptions, TrainingSession

# グローバルにロードしておくと子プロセスが継承できる
TOKENIZER = SentencePieceProcessor()
TOKENIZER.Load(str(BUILD_DIR / "tokenizer.model"))


def build_tokenize_batch(column: str):
    def tokenize_batch(examples):
        ids = []
        num_tokens = []

        for text in examples[column]:
            tokens = TOKENIZER.EncodeAsIds(text)
            num_tokens.append(len(tokens))
            ids.append(tokens)

        return {"input_ids": ids, "num_tokens": num_tokens}

    return tokenize_batch


class BaseTrainer(ABC):
    config: GPTConfig
    tokenizer: SentencePieceProcessor = TOKENIZER
    options: TrainingOptions

    def __init__(
        self,
        /,
        options: TrainingOptions,
    ) -> None:
        self.options = options

        self.config = GPTConfig(
            vocab_size=self.tokenizer.vocab_size(),
            ctx_len=options.ctx_length,
            n_embed=options.n_embed,
            n_heads=options.n_heads,
            n_layers=options.n_layers,
            mlp_ratio=options.mlp_ratio,
        )

    @abstractmethod
    def name(self) -> str: ...

    def load_dataset(self) -> Dataset:
        datasets: list[Dataset] = []

        for dataset_config in DATASET_LIST:
            dataset = load_dataset(
                dataset_config.path,
                dataset_config.name,
                split=dataset_config.train_split,
                cache_dir=str(DATASET_CACHE_DIR),
            )

            # Tokenize (batched & parallel)
            dataset = dataset.map(
                build_tokenize_batch(dataset_config.content_column),
                remove_columns=[dataset_config.content_column],
                batched=True,
                num_proc=os.cpu_count(),  # type: ignore
                desc="Tokenizing dataset (batched & parallel)",  # type: ignore
            )

            # 明示的に Dataset 型にキャスト
            dataset = cast(Dataset, dataset)
            datasets.append(dataset)

        # 連結して共通前処理
        dataset = concatenate_datasets(datasets)

        # Filter sequences by length
        seq_len = self.config.ctx_len + 1
        dataset = dataset.filter(
            lambda ex: len(ex["input_ids"]) >= seq_len,
            num_proc=os.cpu_count(),  # type: ignore
            desc="Filtering by sequence length",  # type: ignore
        )

        # 切り出し：先頭から ctx_len+1 トークンを使う
        # 余計なカラムを削除
        def crop(ex: dict[str, Any]) -> dict[str, Any]:
            ex["ids"] = ex["input_ids"][:seq_len]
            return ex

        dataset = dataset.map(
            crop,
            remove_columns=["input_ids"],
            num_proc=os.cpu_count(),  # type: ignore
            desc="Cropping dataset (batched & parallel)",  # type: ignore
        )

        dataset.set_format(type="torch", columns=["ids"])

        return dataset

    def train(self, session: TrainingSession, *, measure_time: bool = False) -> None:
        # Save hyperparameters in JSON format
        with open(os.path.join(session.options.model_dir, "hparams.json"), "w") as f:
            json.dump(dataclasses.asdict(self.config), f, indent=4)

        self._train(session, measure_time=measure_time)

    @abstractmethod
    def _train(
        self, session: TrainingSession, *, measure_time: bool = False
    ) -> None: ...

    @abstractmethod
    def generate_text(
        self,
        prompt: str,
        *,
        max_new_length: int,
    ) -> str: ...
