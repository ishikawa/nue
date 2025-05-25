# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Nue is a homebrew-scale Large Language Model project for learning transformer architectures. It supports both PyTorch and MLX backends.

## Essential Commands

### Development Commands

- `make check` - Run all checks (lint, type-check, test)
- `make format` - Format code with ruff
- `make lint` - Lint code with ruff
- `make type-check` - Type check with pyright
- `make test` - Run all tests
- `pytest nue/path/to/test.py::TestClass::test_method` - Run a single test

### Training Commands

- `make train-local` - Quick local training with PyTorch (for testing)
- `make train-local-mlx` - Quick local training with MLX
- `make train` - Full training run
- `python -m nue.cli build-corpus` - Build training corpus
- `python -m nue.cli train-tokenizer` - Train tokenizer

## Architecture Overview

### Multi-Backend Design

The project supports two backends with shared interfaces:

- **PyTorch**: Primary backend in `nue/model/torch.py` and `nue/train/torch.py`
- **MLX**: Apple Silicon optimized backend in `nue/mlx/`

When modifying model architecture, ensure changes are reflected in both backends.

### Core Components

1. **Model Implementation**

   - Base configs in `nue/model/base.py` (GPTConfig)
   - PyTorch implementation in `nue/model/torch.py`
   - MLX implementation in `nue/mlx/model.py`
   - Uses Rotary Position Embeddings (RoPE) and standard transformer architecture

2. **Training Pipeline**

   - Dataset handling in `nue/train/dataset.py` (PreTrainingDataset)
   - Trainer base class in `nue/train/trainer.py`
   - Backend-specific trainers inherit from base trainer
   - Supports gradient accumulation, warmup, and checkpoint saving

3. **Data Processing**
   - Corpus building from Wikipedia and livedoor news in `nue/corpus.py`
   - SentencePiece tokenizer (Unigram model, 32K vocab) in `nue/train/tokenizer.py`
   - HuggingFace datasets integration in `nue/datasets.py`

### Key Patterns

- Configuration dataclasses for all major components
- Abstract base classes for backend-agnostic interfaces
- Parallel data loading with multiprocessing
- Time profiling built into training loop
- Checkpoint saving/resuming with optimizer state

### Testing Approach

All major components have corresponding test files (\*\_test.py). When adding new functionality:

- Add tests in the corresponding test file
- Use pytest fixtures for common setup
- Test both PyTorch and MLX implementations when applicable
