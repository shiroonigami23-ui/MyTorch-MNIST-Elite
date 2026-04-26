# Kaggle Runbook (R + HF)

## 1) Create Kaggle Notebook
- Environment: R
- Accelerator: None for lightweight baseline, GPU optional later

## 2) Add Project Files
- Upload this repo as a Kaggle dataset, or clone directly in notebook shell.
- Ensure `scripts/train_mnist_lightweight_kaggle.R` is available.

## 3) Install packages in Kaggle R session
```r
install.packages(c("keras", "jsonlite"))
```

## 4) Train lightweight model
```r
source("scripts/train_mnist_lightweight_kaggle.R")
```

This script writes:
- `checkpoints/mnist_lightweight_mlp.rds`
- `outputs/metrics.json`
- `outputs/history.csv`
- `outputs/confusion_matrix.csv`
- `visuals/r_training_curve.png`

## 5) Generate benchmark report
```bash
python scripts/generate_results_report.py
```

## 6) Upload checkpoint to Hugging Face
Use the Python helper after training:
```bash
python scripts/hf_checkpoint.py upload \
  --checkpoint checkpoints/mnist_lightweight_mlp.rds \
  --repo-id ShiroOnigami23/MyTorch-MNIST \
  --path-in-repo checkpoints/mnist_lightweight_mlp.rds
```

## 7) Deploy/refresh HF Space
```bash
python scripts/setup_hf_space.py \
  --space-id ShiroOnigami23/MyTorch-MNIST-Elite-Demo \
  --folder hf_space
```

## 8) Pull checkpoint later
```bash
python scripts/hf_checkpoint.py download \
  --repo-id ShiroOnigami23/MyTorch-MNIST \
  --path-in-repo checkpoints/mnist_lightweight_mlp.rds \
  --output-dir checkpoints
```

