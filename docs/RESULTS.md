# Results

Last updated: 2026-04-26 06:40 UTC

## Latest Run

| Field | Value |
|---|---|
| Run name | kaggle-r-v6-efficient-cnn |
| Runtime | kaggle-r-notebook |
| Device | gpu |
| Epochs | - |
| Train samples | 60000 |
| Test samples | 10000 |
| Test accuracy | 99.25% |
| Test loss | 0.02210 |

## Goal Tracking

| Milestone | Target | Current |
|---|---:|---:|
| Strong baseline | 98.00% | 99.25% |
| Elite milestone | 98.75% | 99.25% |
| Stretch goal | 99.00% | 99.25% |

## Artifacts

- Metrics JSON: `outputs/metrics.json`
- Run metadata: `outputs/run_metadata.json`
- Visual dashboard assets: `visuals/`
- Checkpoints: `checkpoints/`

## Notes

Efficient CNN v2 with separable conv, batchnorm, LR scheduling, and early stopping.

## How to Refresh This Report

```bash
python scripts/generate_results_report.py
```
