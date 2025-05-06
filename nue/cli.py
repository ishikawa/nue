# Copyright 2025 Takanori Ishikawa
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import dataclasses
import json
import os
from datetime import datetime

import click
from sentencepiece import SentencePieceTrainer

from nue.common import BUILD_DIR
from nue.corpus import build_corpus
from nue.train import Epoch, TrainingOptions, TrainingSession


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
    default=str(BUILD_DIR / "corpus.txt"),
    help="Output file",
)
def build_corpus_command(output_file: str):
    with open(output_file, "w") as f:
        build_corpus(f)


@click.command("train-tokenizer")
@click.option(
    "--vocab-size",
    "vocab_size",
    default=32000,
    type=int,
    help="Vocabulary size",
)
@click.option(
    "--output-prefix",
    "output_prefix",
    required=True,
    type=str,
    default=str(BUILD_DIR / "tokenizer"),
    help="Output prefix",
)
@click.option(
    "--corpus",
    "corpus_file",
    required=True,
    type=click.Path(file_okay=True, dir_okay=False, exists=True),
    default=str(BUILD_DIR / "corpus.txt"),
    help="Input corpus file",
)
def train_tokenizer_command(output_prefix: str, corpus_file: str, vocab_size: int):
    # Training options
    # https://github.com/google/sentencepiece/blob/master/doc/options.md
    SentencePieceTrainer.Train(
        input=corpus_file,
        model_prefix=output_prefix,
        vocab_size=vocab_size,
        character_coverage=0.9995,
        model_type="unigram",
        # Use only a subset of sentences and shuffle them to reduce bias
        input_sentence_size=10_000_000,
        shuffle_input_sentence=True,
        # To avoid OOV, use byte encoding for unknown words
        byte_fallback=True,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
    )


@click.command("train")
@click.option("--n-epochs", "n_epochs", default=1, type=int, help="Number of epochs")
@click.option("--batch-size", "batch_size", default=64, type=int, help="Batch size")
@click.option("--ctx-len", "ctx_len", default=256, type=int, help="Context length")
@click.option(
    "--chunk-overlap-len",
    "chunk_overlap_len",
    default=16,
    type=int,
    help="Training data chunk overlap length",
)
@click.option(
    "--n-embed", "n_embed", default=896, type=int, help="Number of embeddings"
)
@click.option("--n-heads", "n_heads", default=14, type=int, help="Number of heads")
@click.option("--n-layers", "n_layers", default=12, type=int, help="Number of layers")
@click.option("--mlp-ratio", "mlp_ratio", default=4, type=int, help="MLP ratio")
@click.option(
    "--lr",
    "lr",
    default=1e-4,
    type=float,
    help="Learning rate",
)
@click.option(
    "--max-warmup-steps",
    "max_warmup_steps",
    default=5_000,
    type=int,
    help="Maximum number of warmup steps",
)
@click.option("--seed", "seed", default=4649, type=int, help="Random seed")
@click.option(
    "--output-path",
    "output_path",
    default=str(BUILD_DIR / "_train"),
    type=click.Path(file_okay=False, dir_okay=True, exists=False),
    help="Output directory path for the trained model",
)
@click.option("--seed", "seed", default=4649, type=int, help="Random seed")
@click.option(
    "--output-path",
    "output_path",
    default=str(BUILD_DIR / "_train"),
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
@click.option(
    "--override-data-size",
    "override_data_size",
    type=str,
    default=None,
    help="Override data size for training. (e.g. 10%')",
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
    ctx_len: int,
    chunk_overlap_len: int,
    n_embed: int,
    n_heads: int,
    n_layers: int,
    mlp_ratio: int,
    seed: int,
    lr: float,
    max_warmup_steps: int,
    log_interval: int,
    save_interval: int,
    output_path: str,
    model_dir: str | None = None,
    measure_time: bool = False,
    override_data_size: str | None = None,
):
    from nue.train import PyTorchTrainer

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
            ctx_len=ctx_len,
            chunk_overlap_len=chunk_overlap_len,
            n_epochs=n_epochs,
            batch_size=batch_size,
            n_embed=n_embed,
            n_heads=n_heads,
            n_layers=n_layers,
            mlp_ratio=mlp_ratio,
            lr=lr,
            seed=seed,
            model_dir=model_dir,
            log_interval=log_interval,
            save_interval=save_interval,
            override_data_size=override_data_size,
            max_warmup_steps=max_warmup_steps,
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
        + f"n_epochs={training_options.n_epochs}, batch_size={training_options.batch_size}, ctx_length={training_options.ctx_len}, "
        + f"n_embed={training_options.n_embed}, n_heads={training_options.n_heads}, "
        + f"n_layers={training_options.n_layers}, mlp_ratio={training_options.mlp_ratio}, "
        + f"seed={training_options.seed}, "
        + f"lr={training_options.lr}",
        fg="white",
    )

    trainer = PyTorchTrainer(training_options)
    trainer.train(
        training_session,
        measure_time=measure_time,
    )


main.add_command(build_corpus_command)
main.add_command(train_tokenizer_command)
main.add_command(train_command)


if __name__ == "__main__":
    main()
