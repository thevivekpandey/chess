#!/bin/bash

# Script to test individual components before running full pipeline
# Useful for debugging and understanding each step

STOCKFISH="/opt/homebrew/bin/stockfish"  # Adjust to your Stockfish path
MODEL="../attempt_17/best_model.pth"      # Adjust to your model path
DEVICE="mps"

echo "Testing individual components..."
echo "================================"
echo ""

# Test 1: Generate small amount of training data
echo "Test 1: Generate 5 games with SF labeling"
echo "Expected time: ~2-3 minutes"
echo ""
python generate_sf_training_data.py \
  --model "$MODEL" \
  --output test_data.jsonl \
  --stockfish "$STOCKFISH" \
  --games 5 \
  --workers 4 \
  --sf-depth 15 \
  --device "$DEVICE"

if [ $? -eq 0 ]; then
    echo "✓ Test 1 passed: Generated test_data.jsonl"

    # Check data
    LINES=$(wc -l < test_data.jsonl)
    echo "  Generated $LINES positions"

    # Show first example
    echo "  First example:"
    head -1 test_data.jsonl | python -m json.tool | head -20
else
    echo "✗ Test 1 failed"
    exit 1
fi

echo ""
echo "================================"
echo ""

# Test 2: Split data
echo "Test 2: Split train/val data"
python << EOF
import json
import random

with open('test_data.jsonl', 'r') as f:
    lines = f.readlines()

random.shuffle(lines)
split = int(len(lines) * 0.9)

with open('test_train.jsonl', 'w') as f:
    f.writelines(lines[:split])

with open('test_val.jsonl', 'w') as f:
    f.writelines(lines[split:])

print(f"✓ Split into {split} train / {len(lines)-split} val examples")
EOF

if [ $? -eq 0 ]; then
    echo "✓ Test 2 passed: Created test_train.jsonl and test_val.jsonl"
else
    echo "✗ Test 2 failed"
    exit 1
fi

echo ""
echo "================================"
echo ""

# Test 3: Train for 1 epoch
echo "Test 3: Train for 1 epoch"
echo "Expected time: ~1-2 minutes"
echo ""
python train_supervised.py \
  --train-data test_train.jsonl \
  --val-data test_val.jsonl \
  --model "$MODEL" \
  --output test_models \
  --epochs 1 \
  --batch-size 64 \
  --device "$DEVICE"

if [ $? -eq 0 ]; then
    echo "✓ Test 3 passed: Trained model saved to test_models/"
else
    echo "✗ Test 3 failed"
    exit 1
fi

echo ""
echo "================================"
echo ""

# Test 4: Evaluate
echo "Test 4: Evaluate model (10 games)"
echo "Expected time: ~30 seconds"
echo ""
python evaluate_vs_stockfish.py \
  --model test_models/model_epoch_1.pth \
  --stockfish "$STOCKFISH" \
  --games 10 \
  --level 5 \
  --device "$DEVICE"

if [ $? -eq 0 ]; then
    echo "✓ Test 4 passed: Evaluation complete"
else
    echo "✗ Test 4 failed"
    exit 1
fi

echo ""
echo "================================"
echo ""
echo "✓ ALL TESTS PASSED!"
echo ""
echo "You can now run the full pipeline with run_example.sh"
echo ""
echo "Cleaning up test files..."
rm -f test_data.jsonl test_train.jsonl test_val.jsonl
rm -rf test_models
echo "✓ Cleanup complete"
