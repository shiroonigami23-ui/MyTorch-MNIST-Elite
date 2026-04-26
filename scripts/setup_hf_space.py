import argparse
import shutil
from pathlib import Path

from huggingface_hub import HfApi


def sync_assets(repo_root: Path, space_folder: Path) -> None:
    assets_dir = space_folder / "assets"
    visuals_dst = assets_dir / "visuals"
    visuals_dst.mkdir(parents=True, exist_ok=True)

    metrics_src = repo_root / "outputs" / "metrics.json"
    metrics_dst = assets_dir / "metrics.json"
    if metrics_src.exists():
        shutil.copy2(metrics_src, metrics_dst)

    visuals_src = repo_root / "visuals"
    copy_names = [
        "accuracy_curve.png",
        "confusion_matrix.png",
        "learned_features.png",
        "weight_distribution.png",
        "final_heatmap.png",
        "r_training_curve.png",
    ]
    for name in copy_names:
        src = visuals_src / name
        if src.exists():
            shutil.copy2(src, visuals_dst / name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create/update HF Space for MyTorch demo")
    parser.add_argument("--space-id", required=True, help="username/space-name")
    parser.add_argument("--folder", default="hf_space", help="Local folder to upload")
    parser.add_argument("--token", default=None)
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--skip-asset-sync", action="store_true")
    args = parser.parse_args()

    space_folder = Path(args.folder).resolve()
    repo_root = Path(__file__).resolve().parents[1]

    if not space_folder.exists():
        raise FileNotFoundError(f"Folder not found: {space_folder}")

    if not args.skip_asset_sync:
        sync_assets(repo_root, space_folder)

    api = HfApi(token=args.token)
    api.create_repo(
        repo_id=args.space_id,
        repo_type="space",
        private=args.private,
        exist_ok=True,
        space_sdk="gradio",
    )
    api.upload_folder(
        repo_id=args.space_id,
        repo_type="space",
        folder_path=str(space_folder),
    )
    print(f"Space updated: https://huggingface.co/spaces/{args.space_id}")


if __name__ == "__main__":
    main()
