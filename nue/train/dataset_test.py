import pytest

from nue.train.dataset import _tokenize_and_chunk


class DummyTokenizer:
    def EncodeAsIds(self, text: str) -> list[int]:
        return list(range(1, len(text) + 1))


def test_short_text_returns_single_chunk():
    tokenizer = DummyTokenizer()
    result = _tokenize_and_chunk("abc", tokenizer, ctx_len=5, overlap_len=1)
    assert result == [[1, 2, 3]]


def test_overlapping_chunks_are_created_correctly():
    tokenizer = DummyTokenizer()
    text = "abcdefghij"
    result = _tokenize_and_chunk(text, tokenizer, ctx_len=4, overlap_len=1)
    assert result == [
        [1, 2, 3, 4],
        [4, 5, 6, 7],
        [7, 8, 9, 10],
        [10],
    ]


def test_invalid_overlap_raises_value_error():
    tokenizer = DummyTokenizer()
    with pytest.raises(ValueError):
        _tokenize_and_chunk("abcdef", tokenizer, ctx_len=4, overlap_len=4)
