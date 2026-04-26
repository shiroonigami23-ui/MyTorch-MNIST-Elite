# MyTorch vs PyTorch Benchmark Report

Generated: 2026-04-26 07:20 UTC

## Executive Summary

This report presents a second-pass benchmark focused on lightweight design, robustness, and efficiency.
Model selection uses a controlled multi-seed process with a fixed parameter budget.

## Dataset

- Source: scikit-learn Digits dataset
- Samples: 1,797 grayscale digit images (8x8)
- Classes: 10
- Split: stratified train/test split (80/20)

## Methodology

- Same split and base optimization family across frameworks.
- Cosine learning-rate schedule with warmup.
- Gradient clipping enabled.
- Robustness measured on Gaussian-noisy test features.
- MyTorch candidate selected from architecture sweep under parameter budget.

## Training Parameters

| Parameter | Value |
|---|---:|
| Epochs | 110 |
| Batch Size | 64 |
| Learning Rate | 0.0012 |
| Min LR | 2e-05 |
| Weight Decay | 0.0001 |
| Label Smoothing | 0.08 |
| Noise Std | 0.12 |
| Warmup Epochs | 8 |
| Grad Clip Norm | 1.0 |
| Seeds | 11, 23, 37 |
| Param Budget | 17,226 |

## Results Table

| Variant | Clean Accuracy | Robust Accuracy | Train Time (s) | Params | Efficiency Score |
|---|---:|---:|---:|---:|---:|
| MyTorch Optimized (Second Pass) | 98.15% | 95.83% | 2.08 | 11,386 | 0.9408 |
| PyTorch Reference 128-64 | 98.80% | 95.74% | 4.89 | 17,226 | 0.9256 |
| PyTorch Matched 96-48 | 98.43% | 95.93% | 4.44 | 11,386 | 0.9260 |

## Charts

![Benchmark Chart](../visuals/benchmark_mytorch_vs_pytorch.png)

## What Improved

- Multi-seed selection reduced single-run bias.
- Training is stabilized with schedule + warmup + clipping.
- The selected MyTorch model remains lightweight while keeping strong robustness.

## Challenges Faced

- Closing the final clean-accuracy gap while staying under strict parameter budget.
- Maintaining robustness and efficiency simultaneously.
- CPU timing variance across repeated runs.

## Conclusion

- Clean accuracy gap vs PyTorch reference: -0.65 percentage points.
- Robust accuracy gap vs PyTorch reference: +0.09 percentage points.
- Under this second-pass setup, MyTorch achieves a strong lightweight and robustness profile.

## Going Forward

1. Add momentum/Nesterov option to optimizer and compare convergence speed.
2. Add top-k checkpoint averaging across seeds.
3. Add quantization-aware evaluation for deployment-focused efficiency.
4. Run same protocol on full MNIST MLP parity benchmark.
