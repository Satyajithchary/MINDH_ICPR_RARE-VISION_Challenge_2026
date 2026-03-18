#!/bin/bash
# ============================================================================
# ICPR 2026 RARE-VISION — Training Launch Script
# ============================================================================
# Usage: bash scripts/train.sh
#
# Before running:
#   1. Update Config.DATASET_ROOT in rare_vision_pipeline_v3_1.py
#   2. Set Config.MODE = "train"
#   3. Ensure GPU is available
# ============================================================================

set -e

echo "================================================"
echo "ICPR 2026 RARE-VISION — Differential BiomedCLIP"
echo "Mode: TRAIN"
echo "================================================"

# Set environment variables for optimal performance
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export CUDA_LAUNCH_BLOCKING=0

# Run training
python rare_vision_pipeline_v3_1.py

echo "================================================"
echo "Training complete. Check output directory for:"
echo "  - checkpoints/best_model.pth"
echo "  - checkpoints/optimal_thresholds.npy"
echo "  - logs/curves.png"
echo "  - curves/ (PR, ROC, metrics)"
echo "================================================"
