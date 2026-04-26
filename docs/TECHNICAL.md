# MyTorch Technical Notes

## 1. Module Abstraction

`mytorch/nn/module.py` defines the base module contract:
- `forward(...)` for inference/training forward path
- `backward(...)` for gradient propagation
- `__call__(...)` to support module invocation style

This keeps custom layers consistent and composable.

## 2. Linear Layer Formulation

Forward:
`Y = XW^T + b`

Gradients:
- `dW = dY^T X`
- `db = sum(dY)`
- `dX = dY W`

## 3. BatchNorm and Dropout

- BatchNorm stabilizes intermediate activations and training dynamics.
- Dropout reduces overfitting by stochastic neuron masking.

## 4. Optimizer Stack

- Adam for adaptive updates.
- SGD for simpler, controlled baselines.
- Scheduler hooks available in `mytorch/optim/scheduler.py`.

## 5. R-based Lightweight Training

`scripts/train_mnist_lightweight_kaggle.R`:
- Uses Keras in R for Kaggle compatibility.
- Trains a compact MLP baseline on MNIST subset.
- Saves metrics, training history, confusion matrix, and model checkpoint.

## 6. Result Publication Flow

1. Train in Kaggle (R script or notebook).
2. Save artifacts to `outputs/` and `checkpoints/`.
3. Upload checkpoint to Hugging Face model repo.
4. Regenerate `docs/RESULTS.md`.
5. Refresh HF Space for public proof.

## 7. Space Architecture

`hf_space/app.py` reads:
- `outputs/metrics.json` for current KPIs
- `visuals/` for charts and qualitative progress

Tabs expose metrics, visuals, and external project links.

## 8. Reproducibility Controls

- Central config file: `configs/project_config.json`
- Run metadata file: `outputs/run_metadata.json`
- Deterministic seed in R training script (`set.seed(23)`)
