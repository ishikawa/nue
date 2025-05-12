from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class TrainingOptions:
    n_epochs: int
    batch_size: int
    ctx_len: int
    # 学習データセットのチャンク間のオーバーラップ長
    chunk_overlap_len: int
    n_embed: int
    n_heads: int
    n_layers: int
    mlp_ratio: int
    seed: int
    lr: float
    max_warmup_steps: int
    log_interval: int
    save_interval: int
    model_dir: str
    override_data_size: Optional[str] = None


@dataclass
class TrainingSession:
    options: TrainingOptions
