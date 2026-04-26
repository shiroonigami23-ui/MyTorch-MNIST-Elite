# MyTorch vs PyTorch Benchmark Report

Generated: 2026-04-26 06:51 UTC

## Executive Summary

This document compares MyTorch and PyTorch under the same configuration. The objective is to measure accuracy and training efficiency with a controlled setup.

## Dataset

- Source: scikit-learn Digits dataset
- Samples: 1,797 grayscale digit images (8x8)
- Classes: 10
- Split: stratified train/test split (80/20)

## Methodology

- Same model shape in both frameworks: `64 -> 128 -> 64 -> 10` with ReLU activations.
- Same optimizer family: AdamW.
- Same training budget: `80` epochs.
- Same batch size: `64`.
- Same regularization settings: weight decay `0.0001`, label smoothing `0.1`.
- Same random seed: `23`.

## Training Parameters

| Parameter | Value |
|---|---:|
| Epochs | 80 |
| Batch Size | 64 |
| Learning Rate | 0.001 |
| Weight Decay | 0.0001 |
| Label Smoothing | 0.1 |
| Seed | 23 |

## Results Table

| Framework | Test Accuracy | Train Time (s) | Parameter Count |
|---|---:|---:|---:|
| MyTorch | 97.50% | 1.63 | 17,226 |
| PyTorch | 98.89% | 2.98 | 17,226 |

## Accuracy and Efficiency Charts

![Benchmark Chart](../visuals/benchmark_mytorch_vs_pytorch.png)

## What Is Better Than the Baseline

- MyTorch provides full transparency across forward and backward flow.
- The optimizer and loss behavior are easy to audit and customize.
- Model internals can be modified without hidden framework abstractions.

## Challenges Faced

- Matching numerical behavior exactly across frameworks requires careful control of initialization and batching.
- CPU-only timing can vary by environment and background load.
- A controlled MLP benchmark does not represent convolutional production workloads.

## Conclusion

- Accuracy gap (MyTorch - PyTorch): -1.39 percentage points.
- Training time gap (MyTorch - PyTorch): -1.35 seconds.

This benchmark establishes a reproducible baseline. It should be repeated after each optimizer, layer, or data pipeline change.

## Going Forward

1. Add CNN-level parity benchmark for MNIST image tensors.
2. Add multiple seeds and report mean plus standard deviation.
3. Add inference latency and memory profiling.
4. Add CI trend tracking to monitor long-term progress.
