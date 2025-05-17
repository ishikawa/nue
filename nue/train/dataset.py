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

import os
import re
from dataclasses import dataclass
from typing import Any, List, Optional, Protocol, Union, cast

import numpy as np
from datasets import Dataset, Features, Value, concatenate_datasets, load_dataset
from datasets import Sequence as DatasetSequence

from nue.common import DATASET_CACHE_DIR
from nue.datasets import DATASET_LIST

from .tokenizer import TOKENIZER


class TokenizerProtocol(Protocol):
    def EncodeAsIds(self, input: str, **kwargs) -> List[int]: ...

    def EncodeAsPieces(self, input: str, **kwargs) -> List[str]: ...

    def DecodeIds(
        self, input: Union[List[int], np.ndarray], out_type: type = str, **kwargs: Any
    ) -> str: ...

    def GetPieceSize(self) -> int: ...


def _tokenize_and_chunk(
    text: str, tokenizer: TokenizerProtocol, ctx_len: int, overlap_len: int
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
def _map_tokenize_to_chunk_lists(
    examples: dict[str, list], column: str, ctx_len: int, overlap_len: int
) -> dict[str, list]:
    """
    各テキストをチャンク化し、チャンクのリストと、各チャンクの長さのリストを返す。
    出力の行数は入力と同じ。
    """
    output = {"input_ids_chunks": [], "num_tokens_chunks": []}
    # TOKENIZER はグローバル変数から参照
    for text in examples[column]:
        chunks = _tokenize_and_chunk(text, TOKENIZER, ctx_len, overlap_len)
        output["input_ids_chunks"].append(chunks)
        output["num_tokens_chunks"].append([len(c) for c in chunks])
    return output


def load_train_dataset(
    *, ctx_len: int, chunk_overlap_len: int, override_data_size: Optional[str] = None
) -> tuple[
    Dataset, int  # total_tokens
]:
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
        dataset_name = dataset_config.name or dataset_config.path

        # Tokenize
        mapped_dataset = dataset.map(
            _map_tokenize_to_chunk_lists,
            fn_kwargs={
                "column": dataset_config.content_column,
                "ctx_len": ctx_len,
                "overlap_len": chunk_overlap_len,
            },
            # 元のカラムを削除
            remove_columns=dataset.column_names,  # type: ignore
            batched=True,
            num_proc=os.cpu_count(),  # type: ignore
            desc=f"Tokenizing and chunking dataset for '{dataset_name}'",  # type: ignore
        )

        # チャンクリストが空の行を除外 (元のテキストが空など)
        filtered_dataset = mapped_dataset.filter(
            lambda example: len(example["input_ids_chunks"]) > 0,
            num_proc=os.cpu_count(),
            desc=f"Filtering empty rows for '{dataset_name}'",
        )

        # --- 第2段階: フラット化 ---
        # フラット化後のデータセットのスキーマ (特徴量) を定義
        new_features = Features(
            {
                "input_ids": DatasetSequence(Value(dtype="int32")),
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
            desc=f"Flattening chunks for '{dataset_name}' (parallel)",
        )

        processed_dataset = cast(Dataset, flat_dataset)
        datasets_list.append(processed_dataset)

    final_dataset = concatenate_datasets(datasets_list)

    # 合計トークン数を計算
    total_tokens = sum(final_dataset["num_tokens"])

    final_dataset = final_dataset.remove_columns("num_tokens")

    return final_dataset, total_tokens


@dataclass(frozen=True)
class TrainAndValidationDatasetResult:
    total_tokens: int
    train_dataset: Dataset
    validation_dataset: Dataset
