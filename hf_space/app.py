import argparse
import json
from pathlib import Path

import gradio as gr

ROOT = Path(__file__).resolve().parent
METRICS_PATH = ROOT / "assets" / "metrics.json"
VISUALS_DIR = ROOT / "assets" / "visuals"


def load_metrics() -> dict:
    if not METRICS_PATH.exists():
        return {
            "test_accuracy": None,
            "test_loss": None,
            "epochs": None,
            "batch_size": None,
            "train_samples": None,
            "test_samples": None,
        }
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


def metrics_markdown(metrics: dict) -> str:
    acc = metrics.get("test_accuracy")
    loss = metrics.get("test_loss")
    acc_text = f"{acc * 100:.2f}%" if isinstance(acc, (int, float)) else "Not available"
    loss_text = f"{loss:.5f}" if isinstance(loss, (int, float)) else "Not available"

    return f"""
## Live Metrics Snapshot

- Test Accuracy: **{acc_text}**
- Test Loss: **{loss_text}**
- Epochs: **{metrics.get('epochs', 'N/A')}**
- Batch Size: **{metrics.get('batch_size', 'N/A')}**
- Train Samples: **{metrics.get('train_samples', 'N/A')}**
- Test Samples: **{metrics.get('test_samples', 'N/A')}**

This Space is a public dashboard for MyTorch-MNIST-Elite experiments.
"""


def load_gallery_images() -> list[tuple[str, str]]:
    candidates = [
        "accuracy_curve.png",
        "confusion_matrix.png",
        "learned_features.png",
        "weight_distribution.png",
        "final_heatmap.png",
        "r_training_curve.png",
    ]
    items: list[tuple[str, str]] = []
    for name in candidates:
        path = VISUALS_DIR / name
        if path.exists():
            items.append((str(path), name))
    return items


def build_app() -> gr.Blocks:
    metrics = load_metrics()
    gallery_items = load_gallery_images()

    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown("# MyTorch-MNIST-Elite Demo")
        gr.Markdown(
            "Custom lightweight framework progress tracker with Kaggle-R training and Hugging Face checkpointing."
        )

        with gr.Tab("Metrics"):
            md = gr.Markdown(metrics_markdown(metrics))
            refresh_btn = gr.Button("Refresh Metrics")

            def refresh() -> str:
                return metrics_markdown(load_metrics())

            refresh_btn.click(fn=refresh, outputs=md)

        with gr.Tab("Visuals"):
            if gallery_items:
                gr.Gallery(value=gallery_items, columns=2, height=420, show_label=False)
            else:
                gr.Markdown("No visuals found yet. Add files to `visuals/` and run setup script.")

        with gr.Tab("Project Links"):
            gr.Markdown(
                """
- GitHub: https://github.com/shiroonigami23-ui/MyTorch-MNIST-Elite
- HF Model: https://huggingface.co/ShiroOnigami23/MyTorch-MNIST
- Docs: `docs/RESULTS.md`, `docs/SHOWCASE.md`
                """
            )

    return demo


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    app = build_app()
    app.launch(server_name=args.host, server_port=args.port)


if __name__ == "__main__":
    main()
