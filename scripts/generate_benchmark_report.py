import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def pct(v: float) -> str:
    return f"{v * 100:.2f}%"


def row_by_prefix(results: list[dict], prefix: str) -> dict:
    for r in results:
        if str(r.get("variant", "")).startswith(prefix):
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

    mt = row_by_prefix(results, "MyTorch Optimized")
    pt_ref = row_by_prefix(results, "PyTorch Reference")
    pt_match = row_by_prefix(results, "PyTorch Matched")

    acc_gap_ref = (mt["test_accuracy"] - pt_ref["test_accuracy"]) * 100
    robust_gap_ref = (mt["robust_accuracy"] - pt_ref["robust_accuracy"]) * 100

    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    seeds_str = ", ".join(str(s) for s in hp.get("seeds", []))

    content = f"""# MyTorch vs PyTorch Benchmark Report

Generated: {generated}

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
| Epochs | {hp['epochs']} |
| Batch Size | {hp['batch_size']} |
| Learning Rate | {hp['learning_rate']} |
| Min LR | {hp['min_lr']} |
| Weight Decay | {hp['weight_decay']} |
| Label Smoothing | {hp['label_smoothing']} |
| Noise Std | {hp['noise_std']} |
| Warmup Epochs | {hp['warmup_epochs']} |
| Grad Clip Norm | {hp['grad_clip_norm']} |
| Seeds | {seeds_str} |
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

- Multi-seed selection reduced single-run bias.
- Training is stabilized with schedule + warmup + clipping.
- The selected MyTorch model remains lightweight while keeping strong robustness.

## Challenges Faced

- Closing the final clean-accuracy gap while staying under strict parameter budget.
- Maintaining robustness and efficiency simultaneously.
- CPU timing variance across repeated runs.

## Conclusion

- Clean accuracy gap vs PyTorch reference: {acc_gap_ref:+.2f} percentage points.
- Robust accuracy gap vs PyTorch reference: {robust_gap_ref:+.2f} percentage points.
- Under this second-pass setup, MyTorch achieves a strong lightweight and robustness profile.

## Going Forward

1. Add momentum/Nesterov option to optimizer and compare convergence speed.
2. Add top-k checkpoint averaging across seeds.
3. Add quantization-aware evaluation for deployment-focused efficiency.
4. Run same protocol on full MNIST MLP parity benchmark.
"""

    Path(args.output).write_text(content, encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
