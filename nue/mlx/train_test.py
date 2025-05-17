import pytest
from datasets import Dataset
from nue.mlx.train import MlxTrainer
from nue.train.base import TrainingOptions

mx = pytest.importorskip("mlx.core")


class DummyTrainer(MlxTrainer):
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
        framework="mlx",
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

    class DummyModel:
        def __init__(self, vocab_size: int):
            self.vocab_size = vocab_size

        def __call__(self, input_ids: mx.array) -> mx.array:
            # Return deterministic logits for testing
            logits = mx.zeros(
                (1, input_ids.shape[1], self.vocab_size), dtype=mx.float32
            )
            logits[0, -1, 0] = 0.1
            logits[0, -1, 1] = 0.2
            logits[0, -1, 2] = 0.7
            logits[0, -1, 3] = 0.05
            logits[0, -1, 4] = 0.05
            return logits

    trainer.model = DummyModel(trainer.config.vocab_size)
    trainer.manual_seed(0)
    out = trainer.generate([1], max_new_tokens=1, top_k=2, temperature=1.0)
    assert len(out) == 2
    assert out[-1] in {1, 2}


def test_generate_greedy(tmp_path):
    opts = _make_options(str(tmp_path))
    trainer = DummyTrainer(opts)

    class DummyModel:
        def __init__(self, vocab_size: int):
            self.vocab_size = vocab_size

        def __call__(self, input_ids: mx.array) -> mx.array:
            logits = mx.zeros(
                (1, input_ids.shape[1], self.vocab_size), dtype=mx.float32
            )
            logits[0, -1, 0] = 0.1
            logits[0, -1, 1] = 0.9
            return logits

    trainer.model = DummyModel(trainer.config.vocab_size)
    out = trainer.generate([0], max_new_tokens=1, top_k=None)
    assert out == [0, 1]
