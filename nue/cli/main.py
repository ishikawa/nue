import dataclasses
import json
import os
from datetime import datetime

import click

from nue.train import Epoch, TrainingOptions, TrainingSession

from .corpus import build_corpus


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
def main():
    pass


@click.command("build-corpus")
@click.option(
    "--output",
    "-o",
    "output_file",
    required=True,
    type=click.Path(file_okay=True, dir_okay=False, exists=False),
    default="build/corpus.txt",
    help="Output file",
)
def build_corpus_command(output_file: str):
    with open(output_file, "w") as f:
        build_corpus(f)


@click.command("train-tokenizer")
@click.option(
    "--vocab-size",
    "vocab_size",
    default=128_000,
    type=int,
    help="Vocabulary size",
)
@click.option(
    "--special-unk",
    "special_unk",
    default="<unk>",
    type=str,
    help="Special token for unknown words",
)
@click.option(
    "--output-prefix",
    "output_prefix",
    required=True,
    type=str,
    default="build/bpe",
    help="Output prefix",
)
@click.option(
    "--corpus",
    "corpus_file",
    required=True,
    type=click.Path(file_okay=True, dir_okay=False, exists=True),
    default="build/corpus.txt",
    help="Input corpus file",
)
def train_tokenizer_command(
    *, output_prefix: str, corpus_file: str, vocab_size: int, special_unk: str
):
    # set the following environment variable to enable parallelism
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers

    tokenizer = Tokenizer(
        models.BPE(
            # 理論上 OOV はありえないが、実装上の都合で
            unk_token=special_unk,
        )
    )

    # バイト単位に分割するプリトークナイザーを設定
    # tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)  # type: ignore

    # https://huggingface.co/docs/tokenizers/api/trainers#tokenizers.trainers.BpeTrainer
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,  # type: ignore
        show_progress=True,  # type: ignore
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),  # type: ignore
        # 低頻度 token はバイト分割に任せる
        min_frequency=2,  # type: ignore
        special_tokens=["<s>", "</s>", special_unk, "<pad>"],  # type: ignore
    )

    # 学習実行
    tokenizer.train(files=[corpus_file], trainer=trainer)

    # 学習済みモデルを JSON 形式で保存
    tokenizer.save(output_prefix + ".json")
    click.secho(
        "✅ Byte-level BPE tokenizer saved to " + output_prefix + ".json", fg="green"
    )


@click.command("train")
@click.option("--n-epochs", "n_epochs", default=20, type=int, help="Number of epochs")
@click.option("--batch-size", "batch_size", default=96, type=int, help="Batch size")
@click.option(
    "--ctx-length", "ctx_length", default=256, type=int, help="Context length"
)
@click.option(
    "--n-embed", "n_embed", default=512, type=int, help="Number of embeddings"
)
@click.option("--n-heads", "n_heads", default=8, type=int, help="Number of heads")
@click.option("--n-layers", "n_layers", default=6, type=int, help="Number of layers")
@click.option("--mlp-ratio", "mlp_ratio", default=4, type=int, help="MLP ratio")
@click.option(
    "--lr",
    "lr",
    # 大き過ぎる初期学習率は、1 エポック目で重みが指数的に増大する
    # 3e-4 は GPT-2 系で最も実績のある値
    default=3e-4,
    type=float,
    help="Learning rate",
)
@click.option(
    "--lr-scheduler-patience",
    "lr_scheduler_patience",
    type=int,
    default=3,
    help="Patience for learning rate scheduler",
)
@click.option("--seed", "seed", default=4649, type=int, help="Random seed")
@click.option(
    "--output-path",
    "output_path",
    default="build/_train",
    type=click.Path(file_okay=False, dir_okay=True, exists=False),
    help="Output directory path for the trained model",
)
@click.option(
    "--model-dir",
    "model_dir",
    type=str,
    help="Directory path which contains the model checkpoint",
)
@click.option(
    "--log-interval",
    "log_interval",
    type=int,
    default=100,
    help="Number of training steps between logging.",
)
@click.option(
    "--save-interval",
    "save_interval",
    type=int,
    default=100,
    help="Number of training steps between saving model checkpoints.",
)
# Options below are for debugging and not saved in training session
@click.option(
    "--measure-time",
    "measure_time",
    type=bool,
    default=False,
    is_flag=True,
    help="Measure time for each step.",
)
def train_command(
    n_epochs: int,
    batch_size: int,
    ctx_length: int,
    n_embed: int,
    n_heads: int,
    n_layers: int,
    mlp_ratio: int,
    seed: int,
    lr: float,
    lr_scheduler_patience: int,
    log_interval: int,
    save_interval: int,
    output_path: str,
    model_dir: str | None = None,
    measure_time: bool = False,
):
    from nue.train.torch import PyTorchTrainer

    if model_dir is not None:
        click.secho(f"Resuming training from checkpoint {model_dir}", fg="green")

        # --- Resume training session
        with open(os.path.join(model_dir, "train.json"), "r") as f:
            raw_session = json.load(f)
            training_session = TrainingSession(
                epochs=[Epoch(**e) for e in raw_session["epochs"]],
                options=TrainingOptions(**raw_session["options"]),
            )
            training_options = training_session.options
    else:
        # model_dir = output_path + "YYYYMMDD_hhmmss"
        model_dir = os.path.join(output_path, datetime.now().strftime("%Y%m%d_%H%M%S"))
        os.makedirs(model_dir, exist_ok=True)

        # --- Create a training session
        training_options = TrainingOptions(
            ctx_length=ctx_length,
            n_epochs=n_epochs,
            batch_size=batch_size,
            n_embed=n_embed,
            n_heads=n_heads,
            n_layers=n_layers,
            mlp_ratio=mlp_ratio,
            lr=lr,
            lr_scheduler_patience=lr_scheduler_patience,
            seed=seed,
            model_dir=model_dir,
            log_interval=log_interval,
            save_interval=save_interval,
        )

        training_session = TrainingSession(
            epochs=[],
            options=training_options,
        )
        with open(os.path.join(model_dir, "train.json"), "w") as f:
            json.dump(
                dataclasses.asdict(training_session),
                f,
                indent=4,
            )

    click.secho(
        "Using torch trainer: "
        + f"n_epochs={training_options.n_epochs}, batch_size={training_options.batch_size}, ctx_length={training_options.ctx_length}, "
        + f"n_embed={training_options.n_embed}, n_heads={training_options.n_heads}, "
        + f"n_layers={training_options.n_layers}, mlp_ratio={training_options.mlp_ratio}, "
        + f"seed={training_options.seed}, "
        + f"lr={training_options.lr}, lr_scheduler_patience={training_options.lr_scheduler_patience}",
        fg="white",
    )

    trainer = PyTorchTrainer(training_options)
    trainer.train(training_session, measure_time=measure_time)


main.add_command(build_corpus_command)
main.add_command(train_tokenizer_command)
main.add_command(train_command)


if __name__ == "__main__":
    main()
