param(
  [Parameter(Mandatory = $true)]
  [string]$HfModelRepo,

  [Parameter(Mandatory = $true)]
  [string]$HfSpaceRepo,

  [string]$CheckpointPath = "checkpoints/mnist_lightweight_mlp.rds"
)

Write-Host "[1/4] Generating results report..."
python scripts/generate_results_report.py

Write-Host "[2/4] Uploading checkpoint to Hugging Face model repo..."
python scripts/hf_checkpoint.py upload --checkpoint $CheckpointPath --repo-id $HfModelRepo --path-in-repo $CheckpointPath

Write-Host "[3/4] Syncing and uploading Hugging Face Space..."
python scripts/setup_hf_space.py --space-id $HfSpaceRepo --folder hf_space

Write-Host "[4/4] Done."
Write-Host "Model: https://huggingface.co/$HfModelRepo"
Write-Host "Space: https://huggingface.co/spaces/$HfSpaceRepo"

