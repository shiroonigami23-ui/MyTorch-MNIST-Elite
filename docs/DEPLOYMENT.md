# Deployment Guide

## Prerequisites

- Hugging Face account token (`HF_TOKEN` or `huggingface-cli login`)
- Python dependencies installed
- Latest experiment outputs generated

## 1) Generate result docs

```bash
python scripts/generate_results_report.py
```

## 2) Upload model checkpoint

```bash
python scripts/hf_checkpoint.py upload \
  --checkpoint checkpoints/mnist_lightweight_mlp.rds \
  --repo-id ShiroOnigami23/MyTorch-MNIST \
  --path-in-repo checkpoints/mnist_lightweight_mlp.rds
```

## 3) Deploy or refresh Space

```bash
python scripts/setup_hf_space.py \
  --space-id ShiroOnigami23/MyTorch-MNIST-Elite-Demo \
  --folder hf_space
```

## 4) One command pipeline (PowerShell)

```powershell
./scripts/publish_pipeline.ps1 \
  -HfModelRepo "ShiroOnigami23/MyTorch-MNIST" \
  -HfSpaceRepo "ShiroOnigami23/MyTorch-MNIST-Elite-Demo"
```

