from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from datasets import Dataset

from nue.mlx.model import NueLM
from nue.mlx.train import MLXTrainer, collate_mlx, collate_np
from nue.train.base import TrainingOptions
from nue.train.tokenizer import IGNORE_TOKEN_ID, PAD_TOKEN_ID

mx = pytest.importorskip("mlx.core")


class DummyTrainer(MLXTrainer):
    def train_loop(
        self,
        *,
        log_validation_max_tokens: int,
        measure_time: bool = False,
    ):
        if False:
            yield


def _make_options(tmp_path: str) -> TrainingOptions:
    return TrainingOptions(
        n_epochs=1,
        batch_size=1,
        ctx_len=4,
        chunk_overlap_len=0,
        n_embed=4,
        n_heads=2,
        n_layers=1,
        mlp_ratio=2,
        seed=0,
        lr=0.01,
        max_warmup_steps=1,
        log_interval=1,
        save_interval=1,
        model_dir=tmp_path,
        backend="mlx",
    )


def test_override_base_lr(tmp_path):
    opts = _make_options(str(tmp_path))
    trainer = DummyTrainer(opts)

    ds = Dataset.from_dict({"input_ids": [[1, 2, 3, 4]]})
    trainer.train_dataset = ds
    trainer.validation_dataset = ds
    trainer.on_load_dataset(ds, ds)
    trainer.on_train_prepare()

    trainer._train(
        log_validation_max_tokens=0,
        measure_time=False,
        override_base_lr=0.2,
    )

    assert pytest.approx(trainer.learning_rate, rel=1e-6) == 0.2


def test_generate_topk_sampling(tmp_path):
    opts = _make_options(str(tmp_path))
    trainer = DummyTrainer(opts)

    # 期待されるロジットを定義
    expected_logits = mx.zeros((1, 1, trainer.config.vocab_size), dtype=mx.float32)
    expected_logits[0, 0, 0] = 0.1
    expected_logits[0, 0, 1] = 0.2
    expected_logits[0, 0, 2] = 0.7
    expected_logits[0, 0, 3] = 0.05
    expected_logits[0, 0, 4] = 0.05

    # モックの設定
    class MockModel(NueLM):
        def __init__(self, vocab_size):
            self.vocab_size = vocab_size

        def __call__(self, input_ids, *, attention_mask=None):
            return expected_logits

    # モックを trainer に設定
    trainer.model = MockModel(trainer.config.vocab_size)
    trainer.manual_seed(0)

    # テスト実行
    out = trainer.generate([1], max_new_tokens=1, top_k=2, temperature=1.0)

    # アサーション
    assert len(out) == 2
    assert out[-1] in {1, 2}


def test_generate_greedy(tmp_path):
    opts = _make_options(str(tmp_path))
    trainer = DummyTrainer(opts)

    class DummyModel(NueLM):
        def __init__(self, vocab_size: int):
            self.vocab_size = vocab_size

        def __call__(self, input_ids: Any, *, attention_mask: Any | None = None) -> Any:
            logits = mx.zeros(
                (1, input_ids.shape[1], self.vocab_size), dtype=mx.float32
            )
            logits[0, -1, 0] = 0.1
            logits[0, -1, 1] = 0.9
            return logits

    trainer.model = DummyModel(trainer.config.vocab_size)
    out = trainer.generate([0], max_new_tokens=1, top_k=None)
    assert out == [0, 1]


@pytest.mark.parametrize(
    "input_np",
    [
        # --- ① パディング無し ---
        np.array([[10, 11, 12, 13]], dtype=np.int32),
        # --- ② 末尾 PAD を含む ---
        np.array([[20, 21, PAD_TOKEN_ID, PAD_TOKEN_ID]], dtype=np.int32),
        # --- ③ バッチサイズ >1 かつ長さ混在（pad_to_size 後想定） ---
        np.array(
            [
                [1, 2, 3, PAD_TOKEN_ID, PAD_TOKEN_ID],
                [40, 41, 42, 43, 44],
            ],
            dtype=np.int32,
        ),
    ],
)
def test_collate_correctness(input_np):
    """collate() が (inputs, mask, labels) を正しく返すか"""
    # NumPy → MX
    input_mx = mx.array(input_np, dtype=mx.int32)

    out_ids_mlx, attn_mask_mlx, labels_mlx = collate_mlx(input_mx)
    x = collate_np({"input_ids": input_np})
    out_ids_np, attn_mask_np, labels_np = (
        x["input_ids"],
        x["attention_mask"],
        x["labels"],
    )

    # 1) input_ids はそのまま返る
    assert np.array_equal(np.asarray(out_ids_mlx), input_np)
    assert np.array_equal(out_ids_np, input_np)

    # 2) attention_mask は PAD 位置だけ False
    expected_mask = input_np != PAD_TOKEN_ID
    assert np.array_equal(np.asarray(attn_mask_mlx), expected_mask)
    assert np.array_equal(attn_mask_np, expected_mask)

    # 3) labels は 1 トークン左シフト & PAD→IGNORE
    shifted = np.full_like(input_np, IGNORE_TOKEN_ID)
    shifted[:, :-1] = input_np[:, 1:]
    shifted = np.where(shifted == PAD_TOKEN_ID, IGNORE_TOKEN_ID, shifted)
    assert np.array_equal(np.asarray(labels_mlx), shifted)
    assert np.array_equal(labels_np, shifted)


def test_collate_label_ignore_for_all_pad():
    """全トークン PAD の行でもラベルはすべて IGNORE"""
    input_np = np.full((2, 4), PAD_TOKEN_ID, dtype=np.int32)

    # MLX
    _, _, labels = collate_mlx(mx.array(input_np))
    assert np.all(np.asarray(labels) == IGNORE_TOKEN_ID)

    # NumPy
    x = collate_np({"input_ids": input_np})
    labels = x["labels"]
    assert np.all(labels == IGNORE_TOKEN_ID)
