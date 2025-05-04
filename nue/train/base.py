import dataclasses
import json
import os
from abc import ABC, abstractmethod
from typing import Any, cast

from datasets import Dataset, concatenate_datasets, load_dataset
from sentencepiece import SentencePieceProcessor

from nue.minigpt import GPTConfig

from .models import TrainingOptions, TrainingSession

# グローバルにロードしておくと子プロセスが継承できる
TOKENIZER = SentencePieceProcessor()
TOKENIZER.Load("build/sp16k_unigram.model")


def tokenize_batch(examples):
    ids = [TOKENIZER.EncodeAsIds(text) for text in examples["text"]]
    return {"input_ids": ids}


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

        for dataset in [
            load_dataset("wikimedia/wikipedia", "20231101.ja", split="train"),
            load_dataset("wikimedia/wikipedia", "20231101.en", split="train[:25%]"),
        ]:
            # Tokenize (batched & parallel)
            dataset = dataset.map(
                tokenize_batch,
                remove_columns=["text"],
                batched=True,
                num_proc=os.cpu_count(),  # type: ignore
                desc="Tokenizing dataset (batched & parallel)",  # type: ignore
            )

            # Filter sequences by length
            seq_len = self.config.ctx_len + 1
            dataset = dataset.filter(
                lambda ex: len(ex["input_ids"]) >= seq_len,
                num_proc=os.cpu_count(),  # type: ignore
                desc="Filtering by sequence length",  # type: ignore
            )

            # 切り出し：先頭から ctx_len+1 トークンを使う例
            def crop(ex: dict[str, Any]) -> dict[str, Any]:
                ex["ids"] = ex["input_ids"][:seq_len]
                return ex

            dataset = dataset.map(
                crop,
                remove_columns=["input_ids"],
                num_proc=os.cpu_count(),  # type: ignore
                desc="Cropping dataset (batched & parallel)",  # type: ignore
            )

            # 明示的に Dataset 型にキャスト
            dataset = cast(Dataset, dataset)
            dataset.set_format(type="torch", columns=["ids"])

            datasets.append(dataset)

        dataset = concatenate_datasets(datasets)
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
