import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def pct(v: float) -> str:
    return f"{v * 100:.2f}%"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate benchmark markdown report")
    parser.add_argument("--input", default="outputs/benchmark_results.json")
    parser.add_argument("--output", default="docs/BENCHMARK_REPORT.md")
    args = parser.parse_args()

    data = json.loads(Path(args.input).read_text(encoding="utf-8"))
    hp = data["hyperparameters"]
    results = {r["framework"]: r for r in data["results"]}
    mt = results["MyTorch"]
    pt = results["PyTorch"]

    acc_diff = (mt["test_accuracy"] - pt["test_accuracy"]) * 100
    time_diff = mt["train_time_sec"] - pt["train_time_sec"]

    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    content = f"""# MyTorch vs PyTorch Benchmark Report

Generated: {generated}

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
- Same training budget: `{hp['epochs']}` epochs.
- Same batch size: `{hp['batch_size']}`.
- Same regularization settings: weight decay `{hp['weight_decay']}`, label smoothing `{hp['label_smoothing']}`.
- Same random seed: `{hp['seed']}`.

## Training Parameters

| Parameter | Value |
|---|---:|
| Epochs | {hp['epochs']} |
| Batch Size | {hp['batch_size']} |
| Learning Rate | {hp['learning_rate']} |
| Weight Decay | {hp['weight_decay']} |
| Label Smoothing | {hp['label_smoothing']} |
| Seed | {hp['seed']} |

## Results Table

| Framework | Test Accuracy | Train Time (s) | Parameter Count |
|---|---:|---:|---:|
| MyTorch | {pct(mt['test_accuracy'])} | {mt['train_time_sec']:.2f} | {mt['params']:,} |
| PyTorch | {pct(pt['test_accuracy'])} | {pt['train_time_sec']:.2f} | {pt['params']:,} |

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

- Accuracy gap (MyTorch - PyTorch): {acc_diff:+.2f} percentage points.
- Training time gap (MyTorch - PyTorch): {time_diff:+.2f} seconds.

This benchmark establishes a reproducible baseline. It should be repeated after each optimizer, layer, or data pipeline change.

## Going Forward

1. Add CNN-level parity benchmark for MNIST image tensors.
2. Add multiple seeds and report mean plus standard deviation.
3. Add inference latency and memory profiling.
4. Add CI trend tracking to monitor long-term progress.
"""

    Path(args.output).write_text(content, encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
