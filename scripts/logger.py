
import wandb
import pickle
import os
from huggingface_hub import HfApi

class ExperimentManager:
    def __init__(self, project_name, experiment_name, config):
        # 1. Initialize W&B
        self.run = wandb.init(project=project_name, name=experiment_name, config=config)
        self.best_accuracy = 0.0
        self.project_path = config['project_path']

    def log_metrics(self, epoch, train_loss, val_acc):
        # Log to W&B
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_accuracy": val_acc
        })

    def save_checkpoint(self, model, accuracy, epoch, hf_repo_id=None):
        # If this is our best model yet, save it
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            checkpoint_path = os.path.join(self.project_path, "best_model.pkl")

            # Save locally to Drive
            with open(checkpoint_path, "wb") as f:
                pickle.dump(model, f)
            print(f"⭐ New Best Model saved with {accuracy:.2f}% accuracy!")

            # Optional: Push to Hugging Face
            if hf_repo_id:
                api = HfApi()
                try:
                    api.upload_file(
                        path_or_fileobj=checkpoint_path,
                        path_in_repo=f"best_model_epoch_{epoch}.pkl",
                        repo_id=hf_repo_id,
                        repo_type="model"
                    )
                    print(f"🚀 Model pushed to Hugging Face: {hf_repo_id}")
                except Exception as e:
                    print(f"❌ HF Upload failed: {e}")

    def finish(self):
        wandb.finish()
