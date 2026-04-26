import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def load_json(path: Path, default: dict) -> dict:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8-sig"))


def pct(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value * 100:.2f}%"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate MyTorch results markdown report")
    parser.add_argument("--metrics", type=Path, default=Path("outputs/metrics.json"))
    parser.add_argument("--out", type=Path, default=Path("docs/RESULTS.md"))
    parser.add_argument("--experiment-meta", type=Path, default=Path("outputs/run_metadata.json"))
    args = parser.parse_args()

    metrics = load_json(args.metrics, {})
    meta = load_json(args.experiment_meta, {})

    test_acc = metrics.get("test_accuracy")
    test_loss = metrics.get("test_loss")
    epochs = metrics.get("epochs")
    train_samples = metrics.get("train_samples")
    test_samples = metrics.get("test_samples")

    run_name = meta.get("run_name", "mnist-lightweight-baseline")
    runtime = meta.get("runtime", "kaggle-r")
    device = meta.get("device", "cpu")
    notes = meta.get("notes", "")

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    content = f"""# Results

Last updated: {now}

## Latest Run

| Field | Value |
|---|---|
| Run name | {run_name} |
| Runtime | {runtime} |
| Device | {device} |
| Epochs | {epochs if epochs is not None else '-'} |
| Train samples | {train_samples if train_samples is not None else '-'} |
| Test samples | {test_samples if test_samples is not None else '-'} |
| Test accuracy | {pct(test_acc)} |
| Test loss | {f'{test_loss:.5f}' if isinstance(test_loss, (int, float)) else '-'} |

## Goal Tracking

| Milestone | Target | Current |
|---|---:|---:|
| Strong baseline | 98.00% | {pct(test_acc)} |
| Elite milestone | 98.75% | {pct(test_acc)} |
| Stretch goal | 99.00% | {pct(test_acc)} |

## Artifacts

- Metrics JSON: `outputs/metrics.json`
- Run metadata: `outputs/run_metadata.json`
- Visual dashboard assets: `visuals/`
- Checkpoints: `checkpoints/`

## Notes

{notes if notes else 'Add notes in outputs/run_metadata.json to keep each run reproducible.'}

## How to Refresh This Report

```bash
python scripts/generate_results_report.py
```
"""

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(content, encoding="utf-8")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
