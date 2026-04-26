import pickle
from pathlib import Path
from typing import Any

import wandb
from huggingface_hub import HfApi


class ExperimentManager:
    def __init__(self, project_name: str, experiment_name: str, config: dict[str, Any]):
        self.run = wandb.init(project=project_name, name=experiment_name, config=config)
        self.best_accuracy = 0.0
        self.project_path = Path(config.get("project_path", ".")).resolve()
        self.checkpoint_dir = self.project_path / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def log_metrics(self, epoch: int, train_loss: float, val_acc: float) -> None:
        wandb.log(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_accuracy": val_acc,
            }
        )

    def save_checkpoint(
        self,
        model: Any,
        accuracy: float,
        epoch: int,
        hf_repo_id: str | None = None,
        hf_token: str | None = None,
    ) -> None:
        if accuracy <= self.best_accuracy:
            return

        self.best_accuracy = accuracy
        checkpoint_path = self.checkpoint_dir / "best_model.pkl"

        with checkpoint_path.open("wb") as f:
            pickle.dump(model, f)
        print(f"New best model saved with {accuracy:.2f}% accuracy at {checkpoint_path}")

        if not hf_repo_id:
            return

        api = HfApi(token=hf_token)
        try:
            api.create_repo(repo_id=hf_repo_id, repo_type="model", exist_ok=True)
            api.upload_file(
                path_or_fileobj=str(checkpoint_path),
                path_in_repo=f"checkpoints/best_model_epoch_{epoch}.pkl",
                repo_id=hf_repo_id,
                repo_type="model",
            )
            print(f"Model pushed to Hugging Face repo: {hf_repo_id}")
        except Exception as e:
            print(f"HF upload failed: {e}")

    def finish(self) -> None:
        wandb.finish()
