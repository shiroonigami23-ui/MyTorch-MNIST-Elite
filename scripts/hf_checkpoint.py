import argparse
import json
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download


def upload_checkpoint(checkpoint: Path, repo_id: str, path_in_repo: str, token: str | None) -> None:
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    api.upload_file(
        path_or_fileobj=str(checkpoint),
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type="model",
    )
    print(f"Uploaded {checkpoint} -> hf://{repo_id}/{path_in_repo}")


def download_checkpoint(repo_id: str, path_in_repo: str, output_dir: Path, token: str | None) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    local_path = hf_hub_download(
        repo_id=repo_id,
        filename=path_in_repo,
        repo_type="model",
        token=token,
        local_dir=str(output_dir),
    )
    print(f"Downloaded to: {local_path}")


def list_repo_files(repo_id: str, token: str | None) -> None:
    api = HfApi(token=token)
    files = api.list_repo_files(repo_id=repo_id, repo_type="model")
    print(json.dumps(files, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage MyTorch checkpoints on Hugging Face Hub")
    sub = parser.add_subparsers(dest="command", required=True)

    up = sub.add_parser("upload", help="Upload a checkpoint file")
    up.add_argument("--checkpoint", required=True, type=Path)
    up.add_argument("--repo-id", required=True)
    up.add_argument("--path-in-repo", required=True)
    up.add_argument("--token", default=None, help="HF token (optional if already logged in)")

    down = sub.add_parser("download", help="Download a checkpoint file")
    down.add_argument("--repo-id", required=True)
    down.add_argument("--path-in-repo", required=True)
    down.add_argument("--output-dir", default="checkpoints", type=Path)
    down.add_argument("--token", default=None)

    ls = sub.add_parser("list", help="List files in HF model repo")
    ls.add_argument("--repo-id", required=True)
    ls.add_argument("--token", default=None)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "upload":
        upload_checkpoint(args.checkpoint, args.repo_id, args.path_in_repo, args.token)
    elif args.command == "download":
        download_checkpoint(args.repo_id, args.path_in_repo, args.output_dir, args.token)
    elif args.command == "list":
        list_repo_files(args.repo_id, args.token)
    else:
        parser.error("Unknown command")


if __name__ == "__main__":
    main()
