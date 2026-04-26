# MyTorch Feature Specification

MyTorch-MNIST-Elite is a from-scratch deep learning framework focused on clarity, reproducibility, and measurable results.

## Core Architecture

- Modular neural network components under `mytorch/nn/`.
- Sequential composition for clean forward/backward flow.
- Fully connected stack optimized for MNIST iteration speed.

## Training + Optimization

- Optimizers: Adam and SGD (`mytorch/optim/`).
- Regularization blocks: Dropout and BatchNorm.
- Lightweight training recipe for fast Kaggle turnaround.

## Experiment Tracking

- W&B logging through `scripts/logger.py`.
- Hugging Face checkpoint upload/download/list support through `scripts/hf_checkpoint.py`.
- Generated benchmark docs via `scripts/generate_results_report.py`.

## Kaggle + R Workflow

- R training script: `scripts/train_mnist_lightweight_kaggle.R`.
- Artifacts per run:
  - `outputs/metrics.json`
  - `outputs/history.csv`
  - `outputs/confusion_matrix.csv`
  - `checkpoints/*.keras`

## Public Showcase Layer

- Space app scaffold in `hf_space/`.
- Result and proof docs:
  - `docs/RESULTS.md`
  - `docs/SHOWCASE.md`

## Current Focus

- Push stable 98.75%+ runs with reproducible settings.
- Improve architecture and training schedule toward 99.0%.
- Keep every claim backed by metrics, artifacts, and public links.
