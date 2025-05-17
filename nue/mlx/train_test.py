import pytest

mx = pytest.importorskip("mlx.core")

from datasets import Dataset

from nue.mlx.train import MlxTrainer
from nue.train.base import TrainingOptions


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

