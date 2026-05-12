#!/bin/bash

# Example script to run the supervised learning pipeline WITH FOUNDATION DATA MIXING
# This prevents catastrophic forgetting by mixing 30% foundation data with new SF-labeled data
# Adjust paths according to your setup

# Configuration
STOCKFISH="/opt/homebrew/bin/stockfish"  # Adjust to your Stockfish path
INITIAL_MODEL="../attempt_17/best_model.pth"  # Adjust to your initial model
FOUNDATION_DATA="../attempt_02/training_data_augmented.csv"  # Your 3.6M supervised examples
BASE_DIR="runs/run_01"
FOUNDATION_RATIO=0.3  # 30% foundation, 70% new data
DEVICE="mps"  # Use "cuda" for NVIDIA GPU, "cpu" for CPU

# Check if Stockfish exists
if [ ! -f "$STOCKFISH" ]; then
    echo "Error: Stockfish not found at $STOCKFISH"
    echo "Please download from https://stockfishchess.org/ and update STOCKFISH path"
    exit 1
fi

# Check if initial model exists
if [ ! -f "$INITIAL_MODEL" ]; then
    echo "Error: Initial model not found at $INITIAL_MODEL"
    echo "Please train an initial model or update INITIAL_MODEL path"
    exit 1
fi

# Check if foundation data exists
if [ ! -f "$FOUNDATION_DATA" ]; then
    echo "Warning: Foundation data not found at $FOUNDATION_DATA"
    echo "Continuing without foundation data mixing (may lead to catastrophic forgetting!)"
    FOUNDATION_DATA=""
fi

# Adaptive difficulty (starts at level 0, auto-advances when >60%)
INITIAL_SF_LEVEL=0  # Starting level (will auto-advance)

# Run pipeline
echo "Starting supervised learning pipeline WITH ADAPTIVE DIFFICULTY..."
echo "Stockfish: $STOCKFISH"
echo "Initial model: $INITIAL_MODEL"
echo "Foundation data: $FOUNDATION_DATA"
echo "Foundation ratio: $FOUNDATION_RATIO (${FOUNDATION_RATIO}% foundation in training mix)"
echo "Starting SF level: $INITIAL_SF_LEVEL (auto-advances when win rate > 60%)"
echo "Output directory: $BASE_DIR"
echo ""
echo "ADAPTIVE DIFFICULTY SYSTEM:"
echo "  - Starts at SF level $INITIAL_SF_LEVEL"
echo "  - Evaluates model at current level after each iteration"
echo "  - If win rate > 60%, automatically advances to next level"
echo "  - Continues indefinitely until model plateaus"
echo "  - Promotion system: Only promotes if candidate beats baseline"
echo ""

if [ -z "$FOUNDATION_DATA" ]; then
    # Run without foundation data
    python supervised_pipeline.py \
      --base-dir "$BASE_DIR" \
      --model "$INITIAL_MODEL" \
      --stockfish "$STOCKFISH" \
      --iterations 100 \
      --games 100 \
      --epochs 10 \
      --batch-size 256 \
      --lr 0.001 \
      --sf-depth 20 \
      --sf-multipv 5 \
      --eval-games 100 \
      --initial-level "$INITIAL_SF_LEVEL" \
      --workers 8 \
      --device "$DEVICE"
else
    # Run with foundation data mixing
    python supervised_pipeline.py \
      --base-dir "$BASE_DIR" \
      --model "$INITIAL_MODEL" \
      --stockfish "$STOCKFISH" \
      --foundation-data "$FOUNDATION_DATA" \
      --foundation-ratio "$FOUNDATION_RATIO" \
      --iterations 100 \
      --games 100 \
      --epochs 10 \
      --batch-size 256 \
      --lr 0.001 \
      --sf-depth 20 \
      --sf-multipv 5 \
      --eval-games 100 \
      --initial-level "$INITIAL_SF_LEVEL" \
      --workers 8 \
      --device "$DEVICE"
fi

echo ""
echo "Training complete!"
echo "Results saved to: $BASE_DIR"
echo "Best model: $BASE_DIR/models/model_best.pth"
echo "Training log: $BASE_DIR/logs/training_log.json"
