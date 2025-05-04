"""
Generate a corpus for training SentencePiece
"""

from typing import IO, Generator

from datasets import load_dataset

DATASET_LIST = [
    # (
    #    path,
    #    name,
    #    split
    # )
    # NOTE: 日本語と英語のデータを組み合わせる。
    #       日英 Wikipedia のデータ量に合わせて比率を調整する。
    (
        "wikimedia/wikipedia",
        "20231101.ja",
        "train[:20%]",
    ),
    (
        "wikimedia/wikipedia",
        "20231101.en",
        "train[:5%]",
    ),
]


def __preprocess_text(text: str) -> Generator[str, None, None]:
    # 行ごとに処理
    for line in text.split("\n"):
        line = line.strip()

        # 短すぎる/長すぎる行を除外
        if not (5 <= len(line) <= 300):
            continue

        yield line


def build_corpus_line() -> Generator[str, None, None]:
    for path, name, split in DATASET_LIST:
        dataset = load_dataset(path, name, split=split)

        for item in dataset:
            for line in __preprocess_text(item["text"]):  # type: ignore
                yield line


def build_corpus(output_file: IO[str]):
    for line in build_corpus_line():
        output_file.write(line + "\n")
