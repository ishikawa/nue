from dataclasses import dataclass


@dataclass(frozen=True)
class Epoch:
    epoch: int
    loss: float


@dataclass(frozen=True)
class TrainingOptions:
    n_epochs: int
    batch_size: int
    ctx_length: int
    n_embed: int
    n_heads: int
    n_layers: int
    mlp_ratio: int
    seed: int
    lr: float
    lr_scheduler_patience: int
    log_interval: int
    save_interval: int
    model_dir: str


@dataclass
class TrainingSession:
    epochs: list[Epoch]
    options: TrainingOptions
