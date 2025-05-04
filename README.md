# nue

![workflow](https://github.com/ishikawa/nue/actions/workflows/nue.yml/badge.svg)

> Homebrew small-scale LLM based on GPT-2

I'd like to gain practical experience with transformers, particularly by understanding their architecture and real-world applications, with a focus on small-scale LLMs. To achieve this, I decided to create _a tiny LLM_. First, I plan to study [excellent articles and papers](#References) to understand the basic concepts and architecture. Next, I will build and improve _my own GPT model_. My goal is to integrate it into web applications, games, and iOS apps that interest me.

## Goal

Build a small language model that generates grammatically correct sentences.

## Philosophy

Keep it simple.

## Approach

### Dataset

- 日本語と英語のみに絞る
- 初期の事前学習には生成データを使わない
  - ライセンス的に不明瞭な点が多い
- 主なデータソース
  - Wikipedia
  - Public-Domain の書籍データ

### Tokenizer

- 日英のコーパスを用いて SentencePiece + Unigram で学習
  - `byte_fallback=True` で OOV (語彙外) 回避
- `vocab_size` は 32,000
  - 日常語・専門語・語尾・助詞などを自然に分割できる
  - 未知語対応力、学習安定性、パラメータサイズもバランスが良い

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

## References

- [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [GPT in 60 Lines of NumPy | Jay Mody](https://jaykmody.com/blog/gpt-from-scratch/)
- [Reproducing GPT-2 (124M) in llm.c in 90 minutes for $20 · karpathy/llm.c · Discussion #481](https://github.com/karpathy/llm.c/discussions/481)
- [Physics of Language Models](https://physics.allen-zhu.com/home)
