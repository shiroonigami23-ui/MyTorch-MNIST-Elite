import argparse
import json
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


def batches(x: np.ndarray, y: np.ndarray, batch_size: int):
    idx = np.random.permutation(len(x))
    for i in range(0, len(x), batch_size):
        b = idx[i : i + batch_size]
        yield x[b], y[b]


def mytorch_predict_proba(model: Sequential, x: np.ndarray) -> np.ndarray:
    logits = model(x)
    exp = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return exp / np.sum(exp, axis=1, keepdims=True)


def run_mytorch(x_train, y_train, x_test, y_test, epochs, batch_size, lr, weight_decay, smoothing):
    model = Sequential(
        Linear(64, 128),
        ReLU(),
        Linear(128, 64),
        ReLU(),
        Linear(64, 10),
    )
    criterion = CrossEntropyLoss(smoothing=smoothing)
    optim = Adam(model.layers, lr=lr, weight_decay=weight_decay)

    t0 = time.perf_counter()
    for _ in range(epochs):
        for xb, yb in batches(x_train, y_train, batch_size):
            logits = model(xb)
            _ = criterion.forward(logits, yb.reshape(-1, 1))
            grad = criterion.backward()
            model.backward(grad)
            optim.step()
            optim.zero_grad()
    train_seconds = time.perf_counter() - t0

    test_probs = mytorch_predict_proba(model, x_test)
    test_pred = np.argmax(test_probs, axis=1)
    test_acc = float(np.mean(test_pred == y_test))

    params = 0
    for layer in model.layers:
        if hasattr(layer, "W"):
            params += int(np.prod(layer.W.shape) + np.prod(layer.b.shape))

    return {
        "framework": "MyTorch",
        "test_accuracy": test_acc,
        "train_time_sec": float(train_seconds),
        "params": params,
        "notes": "NumPy-based MLP with AdamW-style optimizer and label smoothing.",
    }


def run_pytorch(x_train, y_train, x_test, y_test, epochs, batch_size, lr, weight_decay, smoothing):
    device = torch.device("cpu")

    model = nn.Sequential(
        nn.Linear(64, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10),
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    x_train_t = torch.tensor(x_train, dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y_train, dtype=torch.long, device=device)
    x_test_t = torch.tensor(x_test, dtype=torch.float32, device=device)
    y_test_t = torch.tensor(y_test, dtype=torch.long, device=device)

    t0 = time.perf_counter()
    model.train()
    for _ in range(epochs):
        idx = torch.randperm(x_train_t.shape[0], device=device)
        for i in range(0, x_train_t.shape[0], batch_size):
            b = idx[i : i + batch_size]
            xb = x_train_t[b]
            yb = y_train_t[b]
            logits = model(xb)
            loss = F.cross_entropy(logits, yb, label_smoothing=smoothing)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    train_seconds = time.perf_counter() - t0

    model.eval()
    with torch.no_grad():
        pred = torch.argmax(model(x_test_t), dim=1)
        test_acc = float((pred == y_test_t).float().mean().item())

    params = int(sum(p.numel() for p in model.parameters()))

    return {
        "framework": "PyTorch",
        "test_accuracy": test_acc,
        "train_time_sec": float(train_seconds),
        "params": params,
        "notes": "Reference MLP baseline with AdamW and label smoothing.",
    }


def save_visuals(results: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = [r["framework"] for r in results]
    acc = [r["test_accuracy"] * 100 for r in results]
    tsec = [r["train_time_sec"] for r in results]

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("MyTorch vs PyTorch (Same Dataset and Hyperparameters)", fontsize=14, fontweight="bold")

    ax[0].bar(labels, acc, color=["#1f77b4", "#ff7f0e"])
    ax[0].set_ylabel("Test Accuracy (%)")
    ax[0].set_ylim(85, 100)
    ax[0].grid(axis="y", alpha=0.25)
    for i, v in enumerate(acc):
        ax[0].text(i, v + 0.2, f"{v:.2f}%", ha="center", fontsize=10)

    ax[1].bar(labels, tsec, color=["#1f77b4", "#ff7f0e"])
    ax[1].set_ylabel("Training Time (sec)")
    ax[1].grid(axis="y", alpha=0.25)
    for i, v in enumerate(tsec):
        ax[1].text(i, v + max(tsec) * 0.02, f"{v:.2f}s", ha="center", fontsize=10)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(out_dir / "benchmark_mytorch_vs_pytorch.png", dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run apples-to-apples MyTorch vs PyTorch benchmark")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=23)
    args = parser.parse_args()

    set_seed(args.seed)
    x_train, x_test, y_train, y_test = load_data(seed=args.seed)

    mytorch = run_mytorch(
        x_train,
        y_train,
        x_test,
        y_test,
        args.epochs,
        args.batch_size,
        args.lr,
        args.weight_decay,
        args.label_smoothing,
    )
    pytorch = run_pytorch(
        x_train,
        y_train,
        x_test,
        y_test,
        args.epochs,
        args.batch_size,
        args.lr,
        args.weight_decay,
        args.label_smoothing,
    )

    results = [mytorch, pytorch]
    summary = {
        "dataset": "sklearn_digits",
        "methodology": "same architecture, optimizer family, epochs, and split",
        "hyperparameters": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
            "label_smoothing": args.label_smoothing,
            "seed": args.seed,
        },
        "results": results,
    }

    outputs = Path("outputs")
    outputs.mkdir(exist_ok=True)
    visuals = Path("visuals")
    visuals.mkdir(exist_ok=True)

    (outputs / "benchmark_results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pd.DataFrame(results).to_csv(outputs / "benchmark_results.csv", index=False)
    save_visuals(results, visuals)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
