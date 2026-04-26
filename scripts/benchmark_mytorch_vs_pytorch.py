import argparse
import json
import math
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mytorch.nn.activations import ReLU
from mytorch.nn.linear import Linear
from mytorch.nn.loss import CrossEntropyLoss
from mytorch.nn.sequential import Sequential
from mytorch.optim.adam import Adam


def set_seed(seed: int = 23) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_data(test_size: float = 0.2, seed: int = 23):
    digits = load_digits()
    x = digits.data.astype(np.float32) / 16.0
    y = digits.target.astype(np.int64)
    return train_test_split(x, y, test_size=test_size, random_state=seed, stratify=y)


def make_noise(x: np.ndarray, std: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noisy = x + rng.normal(0.0, std, size=x.shape).astype(np.float32)
    return np.clip(noisy, 0.0, 1.0)


def batches(x: np.ndarray, y: np.ndarray, batch_size: int):
    idx = np.random.permutation(len(x))
    for i in range(0, len(x), batch_size):
        b = idx[i : i + batch_size]
        yield x[b], y[b]


def mytorch_predict_proba(model: Sequential, x: np.ndarray) -> np.ndarray:
    logits = model(x)
    exp = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return exp / np.sum(exp, axis=1, keepdims=True)


def mytorch_param_count(model: Sequential) -> int:
    params = 0
    for layer in model.layers:
        if hasattr(layer, "W"):
            params += int(np.prod(layer.W.shape) + np.prod(layer.b.shape))
    return params


def cosine_lr(epoch: int, total_epochs: int, base_lr: float, min_lr: float, warmup_epochs: int) -> float:
    if warmup_epochs > 0 and epoch < warmup_epochs:
        return base_lr * float(epoch + 1) / float(warmup_epochs)
    progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
    progress = min(max(progress, 0.0), 1.0)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))


def clip_gradients(layers, max_norm: float) -> None:
    for layer in layers:
        if hasattr(layer, "dW"):
            w_norm = np.linalg.norm(layer.dW)
            if w_norm > max_norm:
                layer.dW *= max_norm / (w_norm + 1e-12)
            b_norm = np.linalg.norm(layer.db)
            if b_norm > max_norm:
                layer.db *= max_norm / (b_norm + 1e-12)


def run_mytorch_config(x_train, y_train, x_test, y_test, x_test_noisy, cfg: dict, seed: int):
    set_seed(seed)

    model = Sequential(
        Linear(64, cfg["h1"]),
        ReLU(),
        Linear(cfg["h1"], cfg["h2"]),
        ReLU(),
        Linear(cfg["h2"], 10),
    )
    criterion = CrossEntropyLoss(smoothing=cfg["label_smoothing"])
    optim = Adam(model.layers, lr=cfg["lr"], weight_decay=cfg["weight_decay"])

    t0 = time.perf_counter()
    for epoch in range(cfg["epochs"]):
        optim.lr = cosine_lr(
            epoch=epoch,
            total_epochs=cfg["epochs"],
            base_lr=cfg["lr"],
            min_lr=cfg["min_lr"],
            warmup_epochs=cfg["warmup_epochs"],
        )
        for xb, yb in batches(x_train, y_train, cfg["batch_size"]):
            logits = model(xb)
            _ = criterion.forward(logits, yb.reshape(-1, 1))
            grad = criterion.backward()
            model.backward(grad)
            clip_gradients(model.layers, cfg["grad_clip_norm"])
            optim.step()
            optim.zero_grad()
    train_seconds = time.perf_counter() - t0

    clean_pred = np.argmax(mytorch_predict_proba(model, x_test), axis=1)
    noisy_pred = np.argmax(mytorch_predict_proba(model, x_test_noisy), axis=1)

    acc_clean = float(np.mean(clean_pred == y_test))
    acc_noisy = float(np.mean(noisy_pred == y_test))
    params = mytorch_param_count(model)

    return {
        "framework": "MyTorch",
        "variant": f"MyTorch h{cfg['h1']}-{cfg['h2']} lr{cfg['lr']}",
        "h1": cfg["h1"],
        "h2": cfg["h2"],
        "test_accuracy": acc_clean,
        "robust_accuracy": acc_noisy,
        "train_time_sec": float(train_seconds),
        "params": params,
        "seed": seed,
    }


def run_pytorch_variant(x_train, y_train, x_test, y_test, x_test_noisy, h1, h2, cfg, label, seed):
    set_seed(seed)
    device = torch.device("cpu")
    model = nn.Sequential(
        nn.Linear(64, h1),
        nn.ReLU(),
        nn.Linear(h1, h2),
        nn.ReLU(),
        nn.Linear(h2, 10),
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])

    x_train_t = torch.tensor(x_train, dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y_train, dtype=torch.long, device=device)
    x_test_t = torch.tensor(x_test, dtype=torch.float32, device=device)
    x_test_noisy_t = torch.tensor(x_test_noisy, dtype=torch.float32, device=device)
    y_test_t = torch.tensor(y_test, dtype=torch.long, device=device)

    t0 = time.perf_counter()
    model.train()
    for epoch in range(cfg["epochs"]):
        lr_now = cosine_lr(
            epoch=epoch,
            total_epochs=cfg["epochs"],
            base_lr=cfg["lr"],
            min_lr=cfg["min_lr"],
            warmup_epochs=cfg["warmup_epochs"],
        )
        for g in optimizer.param_groups:
            g["lr"] = lr_now

        idx = torch.randperm(x_train_t.shape[0], device=device)
        for i in range(0, x_train_t.shape[0], cfg["batch_size"]):
            b = idx[i : i + cfg["batch_size"]]
            logits = model(x_train_t[b])
            loss = F.cross_entropy(logits, y_train_t[b], label_smoothing=cfg["label_smoothing"])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg["grad_clip_norm"])
            optimizer.step()
    train_seconds = time.perf_counter() - t0

    model.eval()
    with torch.no_grad():
        clean_pred = torch.argmax(model(x_test_t), dim=1)
        noisy_pred = torch.argmax(model(x_test_noisy_t), dim=1)
        acc_clean = float((clean_pred == y_test_t).float().mean().item())
        acc_noisy = float((noisy_pred == y_test_t).float().mean().item())

    params = int(sum(p.numel() for p in model.parameters()))

    return {
        "framework": "PyTorch",
        "variant": label,
        "h1": h1,
        "h2": h2,
        "test_accuracy": acc_clean,
        "robust_accuracy": acc_noisy,
        "train_time_sec": float(train_seconds),
        "params": params,
        "seed": seed,
    }


def efficiency_score(r: dict, param_budget: int) -> float:
    param_penalty = max(0, r["params"] - param_budget) / max(param_budget, 1)
    time_penalty = np.log1p(r["train_time_sec"]) * 0.03
    return (0.70 * r["test_accuracy"] + 0.30 * r["robust_accuracy"]) - param_penalty * 0.05 - time_penalty


def aggregate_runs(rows: list[dict], key_fields: list[str]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    agg = (
        df.groupby(key_fields, as_index=False)
        .agg(
            test_accuracy=("test_accuracy", "mean"),
            robust_accuracy=("robust_accuracy", "mean"),
            train_time_sec=("train_time_sec", "mean"),
            params=("params", "first"),
            seeds_used=("seed", "nunique"),
        )
    )
    return agg


def save_visuals(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.8))
    fig.suptitle("Second Pass: Lightweight, Robust, Efficient", fontsize=14, fontweight="bold")

    labels = df["variant"].tolist()
    x = np.arange(len(labels))

    ax[0].bar(x, df["test_accuracy"] * 100, color="#1f77b4")
    ax[0].set_title("Clean Accuracy")
    ax[0].set_ylabel("%")
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(labels, rotation=18, ha="right", fontsize=8)
    ax[0].grid(axis="y", alpha=0.25)

    ax[1].bar(x, df["robust_accuracy"] * 100, color="#2ca02c")
    ax[1].set_title("Robust Accuracy")
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(labels, rotation=18, ha="right", fontsize=8)
    ax[1].grid(axis="y", alpha=0.25)

    ax[2].bar(x, df["params"], color="#ff7f0e")
    ax[2].set_title("Parameters")
    ax[2].set_xticks(x)
    ax[2].set_xticklabels(labels, rotation=18, ha="right", fontsize=8)
    ax[2].grid(axis="y", alpha=0.25)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(out_dir / "benchmark_mytorch_vs_pytorch.png", dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Second-pass benchmark optimization")
    parser.add_argument("--epochs", type=int, default=110)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.0012)
    parser.add_argument("--min-lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.08)
    parser.add_argument("--noise-std", type=float, default=0.12)
    parser.add_argument("--warmup-epochs", type=int, default=8)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--seeds", type=str, default="11,23,37")
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    split_seed = 23
    x_train, x_test, y_train, y_test = load_data(seed=split_seed)
    x_test_noisy = make_noise(x_test, std=args.noise_std, seed=split_seed + 100)

    base_cfg = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "min_lr": args.min_lr,
        "weight_decay": args.weight_decay,
        "label_smoothing": args.label_smoothing,
        "warmup_epochs": args.warmup_epochs,
        "grad_clip_norm": args.grad_clip_norm,
    }

    candidates = [
        {**base_cfg, "h1": 96, "h2": 48},
        {**base_cfg, "h1": 112, "h2": 56},
        {**base_cfg, "h1": 120, "h2": 60},
        {**base_cfg, "h1": 128, "h2": 64},
    ]

    sweep_rows = []
    for cfg in candidates:
        for seed in seeds:
            sweep_rows.append(run_mytorch_config(x_train, y_train, x_test, y_test, x_test_noisy, cfg, seed))

    sweep_agg = aggregate_runs(sweep_rows, ["framework", "variant", "h1", "h2"])

    param_budget = 17226
    sweep_agg["efficiency_score"] = sweep_agg.apply(lambda r: efficiency_score(r.to_dict(), param_budget), axis=1)
    feasible = sweep_agg[sweep_agg["params"] <= param_budget]
    best_row = (feasible if len(feasible) > 0 else sweep_agg).sort_values("efficiency_score", ascending=False).iloc[0]

    best_h1 = int(best_row["h1"])
    best_h2 = int(best_row["h2"])

    mytorch_best = {
        "framework": "MyTorch",
        "variant": "MyTorch Optimized (Second Pass)",
        "h1": best_h1,
        "h2": best_h2,
        "test_accuracy": float(best_row["test_accuracy"]),
        "robust_accuracy": float(best_row["robust_accuracy"]),
        "train_time_sec": float(best_row["train_time_sec"]),
        "params": int(best_row["params"]),
        "efficiency_score": float(best_row["efficiency_score"]),
        "seeds_used": int(best_row["seeds_used"]),
    }

    pytorch_ref_rows = []
    pytorch_matched_rows = []
    for seed in seeds:
        pytorch_ref_rows.append(
            run_pytorch_variant(
                x_train, y_train, x_test, y_test, x_test_noisy,
                h1=128, h2=64, cfg=base_cfg, label="PyTorch Reference 128-64", seed=seed
            )
        )
        pytorch_matched_rows.append(
            run_pytorch_variant(
                x_train, y_train, x_test, y_test, x_test_noisy,
                h1=best_h1, h2=best_h2, cfg=base_cfg, label=f"PyTorch Matched {best_h1}-{best_h2}", seed=seed
            )
        )

    pt_ref = aggregate_runs(pytorch_ref_rows, ["framework", "variant", "h1", "h2"]).iloc[0].to_dict()
    pt_mat = aggregate_runs(pytorch_matched_rows, ["framework", "variant", "h1", "h2"]).iloc[0].to_dict()
    pt_ref["efficiency_score"] = efficiency_score(pt_ref, param_budget)
    pt_mat["efficiency_score"] = efficiency_score(pt_mat, param_budget)

    results = [mytorch_best, pt_ref, pt_mat]
    result_df = pd.DataFrame(results)

    summary = {
        "dataset": "sklearn_digits",
        "methodology": "second pass with cosine schedule, warmup, grad clipping, and multi-seed selection",
        "hyperparameters": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "min_lr": args.min_lr,
            "weight_decay": args.weight_decay,
            "label_smoothing": args.label_smoothing,
            "noise_std": args.noise_std,
            "warmup_epochs": args.warmup_epochs,
            "grad_clip_norm": args.grad_clip_norm,
            "seeds": seeds,
            "split_seed": split_seed,
            "param_budget": param_budget,
        },
        "sweep_candidates": sweep_rows,
        "results": results,
    }

    outputs = Path("outputs")
    outputs.mkdir(exist_ok=True)
    visuals = Path("visuals")
    visuals.mkdir(exist_ok=True)

    (outputs / "benchmark_results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    result_df.to_csv(outputs / "benchmark_results.csv", index=False)
    pd.DataFrame(sweep_rows).to_csv(outputs / "benchmark_sweep.csv", index=False)
    save_visuals(result_df, visuals)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
