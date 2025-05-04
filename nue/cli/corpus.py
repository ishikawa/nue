"""
Generate a corpus for training SentencePiece
"""

from typing import IO, Generator

from datasets import load_dataset

from nue.common import DATASET_CACHE_DIR
from nue.datasets import DATASET_LIST


def __preprocess_text(text: str) -> Generator[str, None, None]:
    # 行ごとに処理
    for line in text.split("\n"):
        line = line.strip()

        # 短すぎる行を除外
        if len(line) < 5:
            continue

        yield line


def __build_corpus_line() -> Generator[str, None, None]:
    total = 0
    num_lines_by_lang: dict[str, int] = {}

    for config in DATASET_LIST:
        dataset = load_dataset(
            config.path,
            config.name,
            split=config.tokenizer_split,
            cache_dir=str(DATASET_CACHE_DIR),
        )

        if config.lang not in num_lines_by_lang:
            num_lines_by_lang[config.lang] = 0

        for item in dataset:
            for line in __preprocess_text(item[config.content_column]):  # type: ignore
                num_lines_by_lang[config.lang] += 1
                total += 1
                yield line

    print(f"sum: {total:10}")
    for lang, num_lines in num_lines_by_lang.items():
        print(f"{lang}: {num_lines:10} ({num_lines / total * 100:.2f}%)")


def build_corpus(output_file: IO[str]):
    for line in __build_corpus_line():
        output_file.write(line + "\n")
