import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def pct(v: float) -> str:
    return f"{v * 100:.2f}%"


def row_by_variant(results: list[dict], prefix: str) -> dict:
    for r in results:
        if r["variant"].startswith(prefix):
            return r
    return results[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate benchmark markdown report")
    parser.add_argument("--input", default="outputs/benchmark_results.json")
    parser.add_argument("--output", default="docs/BENCHMARK_REPORT.md")
    args = parser.parse_args()

    data = json.loads(Path(args.input).read_text(encoding="utf-8"))
    hp = data["hyperparameters"]
    results = data["results"]

    mt = row_by_variant(results, "MyTorch Optimized")
    pt_ref = row_by_variant(results, "PyTorch Reference")
    pt_match = row_by_variant(results, "PyTorch Matched")

    acc_gap_ref = (mt["test_accuracy"] - pt_ref["test_accuracy"]) * 100
    robust_gap_ref = (mt["robust_accuracy"] - pt_ref["robust_accuracy"]) * 100

    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    content = f"""# MyTorch vs PyTorch Benchmark Report

Generated: {generated}

## Executive Summary

This report evaluates model quality using clean accuracy, noisy-input robustness, training time, and parameter count.
The MyTorch candidate is selected from a lightweight architecture sweep under a fixed parameter budget.

## Dataset

- Source: scikit-learn Digits dataset
- Samples: 1,797 grayscale digit images (8x8)
- Classes: 10
- Split: stratified train/test split (80/20)

## Methodology

- Same train/test split and seed across runs.
- Same optimizer family (AdamW-style), same batch size, same epochs.
- Robustness check: additive Gaussian noise on test features.
- MyTorch model chosen by efficiency score with parameter budget constraint.

## Training Parameters

| Parameter | Value |
|---|---:|
| Epochs | {hp['epochs']} |
| Batch Size | {hp['batch_size']} |
| Learning Rate | {hp['learning_rate']} |
| Weight Decay | {hp['weight_decay']} |
| Label Smoothing | {hp['label_smoothing']} |
| Noise Std | {hp['noise_std']} |
| Seed | {hp['seed']} |
| Param Budget | {hp['param_budget']:,} |

## Results Table

| Variant | Clean Accuracy | Robust Accuracy | Train Time (s) | Params | Efficiency Score |
|---|---:|---:|---:|---:|---:|
| {mt['variant']} | {pct(mt['test_accuracy'])} | {pct(mt['robust_accuracy'])} | {mt['train_time_sec']:.2f} | {mt['params']:,} | {mt['efficiency_score']:.4f} |
| {pt_ref['variant']} | {pct(pt_ref['test_accuracy'])} | {pct(pt_ref['robust_accuracy'])} | {pt_ref['train_time_sec']:.2f} | {pt_ref['params']:,} | {pt_ref['efficiency_score']:.4f} |
| {pt_match['variant']} | {pct(pt_match['test_accuracy'])} | {pct(pt_match['robust_accuracy'])} | {pt_match['train_time_sec']:.2f} | {pt_match['params']:,} | {pt_match['efficiency_score']:.4f} |

## Charts

![Benchmark Chart](../visuals/benchmark_mytorch_vs_pytorch.png)

## What Improved

- The selected MyTorch model is optimized for a light parameter budget.
- Robustness is measured explicitly instead of only clean accuracy.
- Selection now uses a balanced score rather than one metric.

## Challenges Faced

- Matching numerical behavior exactly across frameworks remains difficult.
- Small datasets can produce small run-to-run variance.
- Efficiency outcomes depend on CPU implementation details.

## Conclusion

- Clean accuracy gap vs PyTorch reference: {acc_gap_ref:+.2f} percentage points.
- Robust accuracy gap vs PyTorch reference: {robust_gap_ref:+.2f} percentage points.
- The current MyTorch candidate is lightweight and competitive, with room for further calibration.

## Going Forward

1. Add momentum or Nesterov variant and compare robustness impact.
2. Add gradient clipping and evaluate stability under stronger noise.
3. Add multi-seed benchmark summary with mean and standard deviation.
4. Add quantized inference test for practical deployment efficiency.
"""

    Path(args.output).write_text(content, encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
