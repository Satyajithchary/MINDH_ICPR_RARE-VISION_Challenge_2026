# Checkpoints

Model checkpoints are not tracked in Git due to file size. To obtain them:

## Option 1: Train from scratch
```bash
# Set Config.MODE = "train" in rare_vision_pipeline_v3_1.py
python rare_vision_pipeline_v3_1.py
```
This produces:
- `best_model.pth` — Best validation mAP checkpoint
- `latest_model.pth` — Most recent epoch checkpoint
- `optimal_thresholds.npy` — Per-class optimized thresholds

## Option 2: Download pretrained (if available)
Check the GitHub Releases page for pre-trained checkpoint downloads.

## Checkpoint Contents

Each `.pth` file contains:
```python
{
    "epoch": int,                    # Completed epoch number
    "model_state_dict": OrderedDict, # Model weights
    "optimizer_state_dict": dict,    # Optimizer state (for resume)
    "scaler_state_dict": dict,       # AMP GradScaler state
    "ema_shadow": dict,              # EMA parameter shadow (used for inference)
    "val_mAP": float,               # Validation mAP at this epoch
    "history": dict,                 # Full training history
    "seed": int,                     # Random seed used
}
```

## Submitted Model Details

- **Best epoch**: 4 (of 5)
- **Validation mAP**: 0.2528
- **Test mAP@0.5**: 0.2456
- **Test mAP@0.95**: 0.2353
- **Seed**: 42
