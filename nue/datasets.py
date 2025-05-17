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

"""Configuration helpers for dataset loading."""

from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class DatasetConfig:
    path: str
    lang: Literal["ja", "en"]
    content_column: str
    # tokenizer 用の split
    tokenizer_split: str
    # 学習用の split
    train_split: str
    name: Optional[str] = None
    trust_remote_code: bool = False


DATASET_LIST = [
    # NOTE: 日本語と英語のデータを組み合わせる。
    #       日英 Wikipedia のデータ量に合わせて比率を調整する。
    DatasetConfig(
        path="wikimedia/wikipedia",
        name="20231101.ja",
        lang="ja",
        content_column="text",
        tokenizer_split="train[:20%]",
        train_split="train",
        trust_remote_code=False,
    ),
    DatasetConfig(
        path="wikimedia/wikipedia",
        name="20231101.en",
        lang="en",
        content_column="text",
        tokenizer_split="train[:10%]",
        train_split="train[:30%]",
        trust_remote_code=False,
    ),
    # livedoor ニュースコーパス
    # https://www.rondhuit.com/download.html
    DatasetConfig(
        path="llm-book/livedoor-news-corpus",
        lang="ja",
        content_column="content",
        tokenizer_split="train",
        train_split="train",
        trust_remote_code=True,
    ),
]
