# Quick Start Guide

## What is this?

This implements **Plan 2**: Supervised learning from Stockfish as a teacher.

Instead of self-play reinforcement learning (which suffered from catastrophic forgetting), we:
1. Play games with the policy network (very fast, ~10ms per move)
2. Label each position with Stockfish's analysis
3. Train the network to imitate Stockfish

## Prerequisites

1. **Stockfish binary** - Download from https://stockfishchess.org/
2. **Initial model checkpoint** - From attempt_17 or train a new one
3. **Python packages**:
   ```bash
   pip install torch chess python-chess tqdm numpy
   ```

## Running the Pipeline

### Option 1: Use the example script (easiest)

1. Edit `run_example.sh` to set your paths:
   ```bash
   STOCKFISH="/path/to/stockfish"
   INITIAL_MODEL="/path/to/model.pth"
   ```

2. Run:
   ```bash
   ./run_example.sh
   ```

### Option 2: Run manually

```bash
python supervised_pipeline.py \
  --base-dir runs/run_01 \
  --model /path/to/initial_model.pth \
  --stockfish /path/to/stockfish \
  --foundation-data /path/to/foundation.csv \
  --foundation-ratio 0.3 \
  --initial-level 0 \
  --iterations 100 \
  --games 100 \
  --epochs 10 \
  --device mps
```

## What to expect

**Adaptive Difficulty System:**
- Training starts at Stockfish level 0
- Automatically advances to next level when win rate > 60%
- Skips easy levels (0-2) immediately if already strong
- Continues indefinitely until model plateaus

Each iteration takes **~30-40 minutes**:
- Data generation: ~20-30 min (100 games, 8 workers, SF depth 20)
- Training: ~5-10 min (10 epochs, 256 batch size)
- Evaluation: ~3-5 min (100 games vs current SF level)

**Timeline:**
- **Levels 0-2**: Skipped immediately (too easy)
- **Levels 3-5**: ~5-15 iterations each to reach 60%
- **Levels 6-7**: ~15-30 iterations each
- **Level 8+**: Model may plateau (can't reach 60%)

## Expected Performance

The system automatically finds your model's strength ceiling:
- **Levels 0-2**: Likely skipped (baseline >60%)
- **Level 3**: First challenge (~5-10 iterations to 60%)
- **Level 4-5**: Progressively harder (~10-20 iterations each)
- **Level 6-7**: Near capacity ceiling (~20-30 iterations each)
- **Level 8+**: May not reach 60% (model plateau)

## Monitoring Progress

Watch the console output for:
1. **Data generation**: Number of positions generated
2. **Training**: Train/val loss (should decrease)
3. **Evaluation**: Win/draw/loss record and score vs Stockfish

Example good output:
```
ITERATION 12/100 - Stockfish Level 3
  [Step 1] Generated 5234 positions from 100 games
  [Step 2] Mixing with foundation data (30%): 7123 total examples
  [Step 3] Train Loss: 2.341 → 2.198 (epoch 10)
  [Step 4] Evaluation:
    Baseline: 0.587 (58.7%)
    Candidate: 0.612 (61.2%)
    ✓ PROMOTED! (+2.5%)
    🎯 Baseline exceeds 60%! Advancing to level 4
```

## Troubleshooting

**Issue**: Data generation is very slow
- **Fix**: Reduce `--sf-depth` to 15 or `--workers` to 4

**Issue**: Out of memory during training
- **Fix**: Reduce `--batch-size` to 128 or 64

**Issue**: Model not improving
- **Fix**: Generate more data (`--games 200`) or train longer (`--epochs 20`)

**Issue**: Stockfish not found
- **Fix**: Install Stockfish and update path in script

## Next Steps

After training:

1. **Compare with self-play**: How does this compare to attempt_17?
2. **Test at different strengths**: Use progressive evaluation
   ```bash
   python evaluate_vs_stockfish.py --model models/model_best.pth --stockfish /path/to/sf --progressive
   ```
3. **Generate more data**: Try 500-1000 games for better coverage
4. **Adjust SF depth**: Try depth 25 for stronger labels (slower)

## File Structure

```
attempt_18/
├── chess_engine.py              # Neural network architecture
├── generate_sf_training_data.py # Data generation
├── train_supervised.py          # Training
├── evaluate_vs_stockfish.py     # Evaluation
├── supervised_pipeline.py       # Full pipeline
├── run_example.sh              # Example script
├── README.md                    # Full documentation
└── QUICKSTART.md               # This file

runs/run_01/                     # Output directory (created by pipeline)
├── data/                        # Training data (.csv files)
├── models/                      # Model checkpoints
│   ├── model_M0.pth            # Initial model
│   ├── model_M1.pth            # After iteration 1
│   ├── model_best.pth          # Best promoted model
│   └── iter1/                  # Detailed checkpoints per iteration
└── logs/
    ├── training_log.json        # Full training history
    └── level_history.json       # Level progression tracking
```

## Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--games` | 100 | Games per iteration (more = better but slower) |
| `--epochs` | 10 | Training epochs (balance overfitting) |
| `--sf-depth` | 20 | Stockfish search depth (higher = stronger but slower) |
| `--sf-multipv` | 5 | Number of candidate moves from SF |
| `--workers` | 8 | Parallel workers for labeling |
| `--batch-size` | 256 | Training batch size |
| `--lr` | 0.001 | Learning rate |

## Questions?

See README.md for detailed documentation, design decisions, and troubleshooting.
