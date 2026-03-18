#!/bin/bash
# ============================================================================
# ICPR 2026 RARE-VISION — Test Inference Script
# ============================================================================
# Usage: bash scripts/test.sh
#
# Before running:
#   1. Ensure best_model.pth exists in checkpoints/
#   2. Update Config.TEST_DATA_ROOT in rare_vision_pipeline_v3_1.py
#   3. Set Config.MODE = "test"
# ============================================================================

set -e

echo "================================================"
echo "ICPR 2026 RARE-VISION — Differential BiomedCLIP"
echo "Mode: TEST INFERENCE"
echo "================================================"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

python rare_vision_pipeline_v3_1.py

echo "================================================"
echo "Inference complete. Outputs:"
echo "  - results/test_predictions.json"
echo "  - results/test_predictions.xlsx"
echo "  - results/test_*_frames.csv"
echo "================================================"
