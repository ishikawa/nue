# nue

![workflow](https://github.com/ishikawa/nue/actions/workflows/nue.yml/badge.svg)

> Homebrew small-scale LLM based on GPT-2

I'd like to gain practical experience with transformers, particularly by understanding their architecture and real-world applications, with a focus on small-scale LLMs. To achieve this, I decided to create _a tiny LLM_. My goal is to integrate it into web applications, games, and iOS apps that interest me.

## Goal

Build a small language model that generates grammatically correct sentences.

## Philosophy

Keep it simple, but not simplistic.

## Dataset

| 名前                                                                              | ライセンス                                                            | 備考                                                                                                          |
| --------------------------------------------------------------------------------- | --------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| [Wikimedia Wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia)        | [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/)       |                                                                                                               |
| [livedoor ニュースコーパス](https://www.rondhuit.com/download.html#news%20corpus) | [CC BY-ND 2.1 JP](https://creativecommons.org/licenses/by-nd/2.1/jp/) | データソースは [llm-book/livedoor-news-corpus](https://huggingface.co/datasets/llm-book/livedoor-news-corpus) |

## Tokenizer

日英のコーパスを用いて SentencePiece + Unigram で学習します。

- `byte_fallback=True` で OOV (語彙外) 回避
- `vocab_size` は 32,000

(1) コーパスを生成

以下のコマンドを実行すると、 `build/corpus.txt` が生成されます。

```
$ poetry run nue build-corpus
```

(2) Tokenizer を学習

以下のコマンドを実行すると、 `build/sp8k_unigram.model` と `build/sp8k_unigram.vocab` が生成されます。

```
$ poetry run nue train-tokenizer
```

## Training

```
poetry run nue train
```

## License

Apache License 2.0

## References

Excellent articles and papers that I've read:

- [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [GPT in 60 Lines of NumPy | Jay Mody](https://jaykmody.com/blog/gpt-from-scratch/)
- [Reproducing GPT-2 (124M) in llm.c in 90 minutes for $20 · karpathy/llm.c · Discussion #481](https://github.com/karpathy/llm.c/discussions/481)
- [Physics of Language Models](https://physics.allen-zhu.com/home)
