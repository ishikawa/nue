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


DATASET_LIST = [
    # NOTE: 日本語と英語のデータを組み合わせる。
    #       日英 Wikipedia のデータ量に合わせて比率を調整する。
    DatasetConfig(
        path="wikimedia/wikipedia",
        name="20231101.ja",
        lang="ja",
        content_column="text",
        tokenizer_split="train[:20%]",
        # train_split="train",
        train_split="train[:1%]",
    ),
    DatasetConfig(
        path="wikimedia/wikipedia",
        name="20231101.en",
        lang="en",
        content_column="text",
        tokenizer_split="train[:10%]",
        # train_split="train[:30%]",
        train_split="train[:1%]",
    ),
    # livedoor ニュースコーパス
    # https://www.rondhuit.com/download.html
    DatasetConfig(
        path="llm-book/livedoor-news-corpus",
        lang="ja",
        content_column="content",
        tokenizer_split="train",
        train_split="train[:1%]",
    ),
]
